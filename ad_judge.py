"""Claude Vision judge for ad/editorial/furniture classification.

Each call receives 1..N cropped PNG bytes. We classify each crop as AD,
EDITORIAL, or FURNITURE using Claude Haiku 4.5 Vision. Results are cached
by SHA256 of the cropped bytes so re-uploads of the same paper hit cache
and pay near-zero API cost.

Usage from app.py:

    from ad_judge import judge_candidates, Verdict, JudgeUnavailable

    try:
        verdicts = judge_candidates(
            crops=[png_bytes_a, png_bytes_b, ...],
            page_context="Page 3 of broadsheet newspaper",
            cache_model=AdJudgeCache,
            db_session=db.session,
        )
    except JudgeUnavailable as e:
        # No API key, or mock-mode-but-anthropic-import-failed.
        # Fall back to persisting raw candidates unjudged.
        ...

The `cache_model` parameter is an SQLAlchemy model class with columns
(crop_hash, verdict, reason, model_used, created_date). It is injected
to avoid circular imports with app.py.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


# Default judging model. Opus 4.8 gives the best accuracy on the nuanced calls
# (obituaries / legal notices that read like news but are paid ads, house ads vs.
# editorial). Override with AD_JUDGE_MODEL to trade accuracy for cost/latency
# (e.g. "claude-sonnet-4-6"). The resolved model is part of the cache key, so
# changing it re-judges crops instead of serving verdicts from the old model.
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"
MAX_IMAGES_PER_CALL = 4
DEFAULT_MAX_CALLS_PER_PAPER = int(os.environ.get("AD_JUDGE_MAX_CALLS_PER_PAPER", "500"))


def resolve_judge_model() -> str:
    """The model used for classification, from AD_JUDGE_MODEL or the default."""
    return os.environ.get("AD_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL

# Bump this string whenever SYSTEM_PROMPT changes so the cache keys off the
# (crop + prompt-version) pair. Old entries become unreachable (no lookup will
# match), but they're harmless and will age out. Never silently reuse verdicts
# from a prior prompt — a reworded rubric can flip AD/EDITORIAL decisions.
PROMPT_VERSION = "v4_2026-06-05_ad_bbox_refinement"

VERDICT_AD = "AD"
VERDICT_EDITORIAL = "EDITORIAL"
VERDICT_FURNITURE = "FURNITURE"
_VALID_VERDICTS = {VERDICT_AD, VERDICT_EDITORIAL, VERDICT_FURNITURE}

SYSTEM_PROMPT = (
    "You are helping measure paid advertisements on a newspaper page.\n"
    "Look at each cropped region and classify it as exactly one of:\n"
    "- AD: a paid advertisement. This includes: display ads, classified display ads,\n"
    "  service-directory tiles, house ads promoting the newspaper's own services,\n"
    "  OBITUARIES / DEATH NOTICES (paid by the family), IN MEMORIAMS / CARDS OF THANKS,\n"
    "  and LEGAL / PUBLIC NOTICES (paid by government bodies, law firms, or individuals):\n"
    "  meeting minutes, ordinances, resolutions, public-hearing and bid/proposal notices,\n"
    "  summons, probate / estate and foreclosure / mortgage notices, delinquent-tax lists,\n"
    "  assumed-name (DBA) filings, official financial / budget statements, election notices.\n"
    "- EDITORIAL: editorial content (news story, feature story, opinion column,\n"
    "  photo with caption, byline, news brief, continuation/jump, staff-written\n"
    "  community news). Editorial has a reporter byline or is staff-written copy.\n"
    "- FURNITURE: page furniture (masthead, section header like CLASSIFIEDS /\n"
    "  BRIEFLY / OBITUARIES as a section label, folio/date strip, navigation\n"
    "  index, weather box, barcode/UPC, paper's own nameplate).\n"
    "\n"
    "Rules of thumb:\n"
    "- Obituaries are ALWAYS paid ads, even though they read like short news items.\n"
    "  A block with a person's name as headline, life dates (e.g. 'Jan. 4, 1923 - Dec. 25, 2022'),\n"
    "  funeral / visitation / survivors info, or 'preceded in death by' language = AD.\n"
    "- LEGAL / PUBLIC NOTICES are PAID ADS, not editorial -- the most commonly missed\n"
    "  case. Treat as AD any block of dense, formal, official text published verbatim:\n"
    "  'NOTICE IS HEREBY GIVEN', statute / section references, city / county / school-board\n"
    "  minutes or ordinances, summons, probate, foreclosure / mortgage, delinquent-tax\n"
    "  lists, bid / quote requests, assumed-name filings, budget / financial statements.\n"
    "  These read like official documents (not a reporter's narrative), usually in small\n"
    "  dense type, often under a 'PUBLIC NOTICES' / 'LEGAL NOTICES' banner.\n"
    "  KEY DISTINCTION: the government's OWN published text (verbatim minutes, ordinance,\n"
    "  summons, financial report) = AD; a reporter's NEWS STORY about a meeting or issue\n"
    "  (narrative prose, byline, dateline, quotes) = EDITORIAL. If a crop is a FRAGMENT of\n"
    "  such official text with no visible header, still judge by its official / legal\n"
    "  character: dense statutory / procedural / tabular government text = AD.\n"
    "- A region with a paid advertiser's name + phone/price/website = AD (including\n"
    "  the newspaper's own house ads for subscriptions, job openings, classified rates).\n"
    "- A region dominated by reporter-written news copy, bylines, datelines, photo\n"
    "  credits, or photo cutlines = EDITORIAL, even if it sits inside a rule border.\n"
    "- A label identifying a section (the word CLASSIFIEDS / BRIEFLY / NEWS / SPORTS\n"
    "  as a banner, the masthead with paper name and date) = FURNITURE.\n"
    "- When a region contains BOTH editorial copy and a clear advertiser block, call\n"
    "  the dominant element by area and visual weight.\n"
    "- A decorative sub-panel of a larger ad (a single headline block, an icon panel,\n"
    "  an inset quote box) that's clearly part of a parent ad = AD (it's still paid\n"
    "  space), but note in REASON that it's a sub-region.\n"
    "- BOUNDING BOX (for AD verdicts): if the crop is a paid ad but ALSO contains\n"
    "  decorative chrome that is NOT part of the ad -- a SHARED section header / banner\n"
    "  spanning a group of ads (e.g. 'Shop Local for Mother's Day', an 'Entertainment'\n"
    "  or 'Dining Guide' feature header), a shaded feature-section background, or a\n"
    "  title strip -- return the TIGHT bounding box of JUST the advertisement so the\n"
    "  shared header/banner is excluded. Give it as four decimals 0-1 of THIS crop:\n"
    "  left,top,right,bottom (e.g. '0.42,0,1,1' = the ad is the right ~58%; '0,0.25,1,1'\n"
    "  = the ad is the bottom 75%, below a top banner). If the ENTIRE crop is the\n"
    "  advertisement, answer 'full'. Only tighten when you are confident the excluded\n"
    "  part is shared/editorial chrome -- NOT the ad's own logo, headline, or border.\n"
    "  When unsure, answer 'full'.\n"
    "\n"
    "For each image in order, respond in this exact format, one stanza per image,\n"
    "stanzas separated by a blank line:\n"
    "IMAGE <n>\n"
    "VERDICT: AD|EDITORIAL|FURNITURE\n"
    "REASON: one short sentence.\n"
    "BOX: full   (or  left,top,right,bottom  -- only meaningful for AD; else 'full')\n"
)


@dataclass
class Verdict:
    verdict: str            # "AD" | "EDITORIAL" | "FURNITURE"
    reason: str
    model_used: str
    cache_hit: bool
    crop_hash: str
    # Tight ad bounds WITHIN the crop, as (left, top, right, bottom) fractions
    # 0..1, when the crop also contains shared header/chrome that isn't the ad.
    # None means "use the whole candidate region" (the common case).
    crop_box: Optional[tuple] = None

    @property
    def is_ad(self) -> bool:
        return self.verdict == VERDICT_AD


class JudgeUnavailable(RuntimeError):
    """Raised when the judge cannot run (no API key, mock-mode misconfig, etc.)."""


def _hash_crop(crop: bytes, model: str) -> str:
    """Hash binds (PROMPT_VERSION, model, crop) so the cache invalidates when
    either the rubric OR the judging model changes. Without the model in the
    key, swapping models would silently keep serving the old model's verdicts."""
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode("utf-8"))
    h.update(b"\0")
    h.update((model or "").encode("utf-8"))
    h.update(b"\0")
    h.update(crop)
    return h.hexdigest()


def _mock_verdict(crop: bytes) -> Verdict:
    """Offline deterministic verdict for AD_JUDGE_MOCK=1 runs.
    Classifies everything as AD so the full pipeline exercises without API calls.
    """
    h = _hash_crop(crop, "mock")
    return Verdict(
        verdict=VERDICT_AD,
        reason="mock-mode (AD_JUDGE_MOCK=1): treating all candidates as AD",
        model_used="mock",
        cache_hit=False,
        crop_hash=h,
    )


def _cached_verdict(crop_hash: str, cache_model, db_session) -> Optional[Verdict]:
    if cache_model is None or db_session is None:
        return None
    try:
        row = cache_model.query.filter_by(crop_hash=crop_hash).first()
    except Exception as e:
        print(f"[ad_judge] cache lookup failed: {e}")
        return None
    if not row:
        return None
    reason, box = _decode_reason(row.reason)
    return Verdict(
        verdict=row.verdict,
        reason=reason,
        model_used=row.model_used or resolve_judge_model(),
        cache_hit=True,
        crop_hash=crop_hash,
        crop_box=box,
    )


def _persist_verdict(v: Verdict, cache_model, db_session) -> None:
    if cache_model is None or db_session is None:
        return
    try:
        existing = cache_model.query.filter_by(crop_hash=v.crop_hash).first()
        if existing:
            return
        row = cache_model(
            crop_hash=v.crop_hash,
            verdict=v.verdict,
            reason=_encode_reason(v.reason, v.crop_box)[:500] if (v.reason or v.crop_box) else None,
            model_used=v.model_used,
            created_date=datetime.utcnow(),
        )
        db_session.add(row)
        db_session.commit()
    except Exception as e:
        print(f"[ad_judge] failed to persist cache row: {e}")
        try:
            db_session.rollback()
        except Exception:
            pass


_IMAGE_RE = re.compile(r"IMAGE\s*(\d+)", re.IGNORECASE)
_VERDICT_RE = re.compile(r"VERDICT:\s*(AD|EDITORIAL|FURNITURE)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*([^\n]+)", re.IGNORECASE)
_BOX_RE = re.compile(r"BOX:\s*([^\n]+)", re.IGNORECASE)
_NUM_RE = re.compile(r"[0-9]*\.?[0-9]+")

# A box this close to the full crop isn't worth tightening; smaller than this
# is almost certainly a parse error or a mis-crop, so ignore it (keep full box).
_BOX_MAX_AREA = 0.92
_BOX_MIN_AREA = 0.04


def _parse_box(val: Optional[str]):
    """Parse a BOX value into (l,t,r,b) fractions, or None for full/invalid.

    Conservative on purpose: returns None (=> keep the whole candidate region)
    unless the model gave a well-formed sub-rectangle that meaningfully — but
    not absurdly — tightens the crop. This guarantees refinement can only ever
    SHRINK a box toward a real sub-region, never enlarge or corrupt it.
    """
    if not val:
        return None
    v = val.strip().lower()
    if not v or v.startswith("full") or v.startswith("none") or v.startswith("n/a"):
        return None
    nums = _NUM_RE.findall(v)
    if len(nums) < 4:
        return None
    try:
        l, t, r, b = (float(x) for x in nums[:4])
    except ValueError:
        return None
    l, t = max(0.0, min(1.0, l)), max(0.0, min(1.0, t))
    r, b = max(0.0, min(1.0, r)), max(0.0, min(1.0, b))
    if r <= l or b <= t:
        return None
    area = (r - l) * (b - t)
    if area >= _BOX_MAX_AREA or area < _BOX_MIN_AREA:
        return None
    return (l, t, r, b)


def parse_response(text: str, expected_count: int):
    """Return a list of (verdict, reason, box) tuples of length expected_count.

    box is (l,t,r,b) fractions of the crop or None. Parser is permissive: if the
    model returns fewer stanzas than expected, missing entries default to
    EDITORIAL (safer: we'd rather drop an ad than persist an editorial false
    positive). Stanzas are split on IMAGE markers so an optional BOX line can't
    leak across images.
    """
    out = [(VERDICT_EDITORIAL, "unparsed: no stanza found", None)
           for _ in range(expected_count)]
    marks = list(_IMAGE_RE.finditer(text))
    for k, m in enumerate(marks):
        try:
            idx = int(m.group(1)) - 1
        except ValueError:
            continue
        if not (0 <= idx < expected_count):
            continue
        chunk = text[m.end(): marks[k + 1].start() if k + 1 < len(marks) else len(text)]
        vm = _VERDICT_RE.search(chunk)
        if not vm:
            continue
        verdict = vm.group(1).upper()
        if verdict not in _VALID_VERDICTS:
            continue
        rm = _REASON_RE.search(chunk)
        reason = rm.group(1).strip() if rm else ""
        box = _parse_box(_BOX_RE.search(chunk).group(1)) if _BOX_RE.search(chunk) else None
        out[idx] = (verdict, reason, box if verdict == VERDICT_AD else None)
    return out


# The crop_box rides along in the cache's reason column (no schema change). The
# tag is appended so a plain reason round-trips unchanged when there's no box.
_BOX_TAG = " ::BOX="


def _encode_reason(reason: str, box) -> str:
    reason = (reason or "")[:440]
    if not box:
        return reason
    l, t, r, b = box
    return f"{reason}{_BOX_TAG}{l:.4f},{t:.4f},{r:.4f},{b:.4f}"


def _decode_reason(stored: Optional[str]):
    if not stored or _BOX_TAG not in stored:
        return (stored or "", None)
    reason, _, tag = stored.partition(_BOX_TAG)
    nums = _NUM_RE.findall(tag)
    if len(nums) < 4:
        return (reason, None)
    try:
        return (reason, _parse_box(",".join(nums[:4])))
    except ValueError:
        return (reason, None)


def _resolve_api_key() -> Optional[str]:
    """Resolve ANTHROPIC_API_KEY with .env fallback.

    On Windows, a blank user-level `ANTHROPIC_API_KEY=` env var can mask
    the value in .env (because `load_dotenv()` does not override existing
    vars). Treat empty as missing and force-load from .env when that happens.
    """
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        from dotenv import dotenv_values
        vals = dotenv_values()
        k = vals.get("ANTHROPIC_API_KEY")
        if k:
            os.environ["ANTHROPIC_API_KEY"] = k
            return k
    except Exception:
        pass
    return None


def _call_claude(crops: List[bytes], page_context: Optional[str], model: str):
    """Issue a single Claude Messages call classifying all crops in the list.

    Returns a list of (verdict, reason) aligned to `crops`. On any API error
    this raises; the caller decides whether to fall back.
    """
    try:
        import anthropic  # local import so mock-mode callers don't need the SDK
    except ImportError as e:
        raise JudgeUnavailable(f"anthropic SDK not installed: {e}")

    if not _resolve_api_key():
        raise JudgeUnavailable("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic()

    content: list = []
    if page_context:
        content.append({"type": "text", "text": f"Context: {page_context}"})
    for i, crop in enumerate(crops, start=1):
        content.append({"type": "text", "text": f"IMAGE {i}"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(crop).decode("ascii"),
            },
        })

    # Retry transient server-side conditions (529 overloaded, 429 rate-limit,
    # 5xx) with exponential backoff. Without this a brief Anthropic overload
    # makes EVERY batch error -> all candidates default to EDITORIAL -> the whole
    # paper comes back with ~0 ads. Backoff: 2,4,8,16s across 5 attempts.
    resp = None
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": content}],
            )
            break
        except Exception as e:
            status = getattr(e, "status_code", None)
            retryable = status in (429, 500, 502, 503, 529) or "overloaded" in str(e).lower()
            if retryable and attempt < 4:
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16
                print(f"[ad_judge] transient API error ({status or 'overloaded'}); "
                      f"retry {attempt + 1}/4 in {wait}s")
                time.sleep(wait)
                continue
            raise
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
    usage = getattr(resp, "usage", None)
    if usage is not None:
        try:
            print(f"[ad_judge] API call: "
                  f"in={getattr(usage,'input_tokens',0)} "
                  f"out={getattr(usage,'output_tokens',0)} "
                  f"cache_read={getattr(usage,'cache_read_input_tokens',0)} "
                  f"cache_write={getattr(usage,'cache_creation_input_tokens',0)}")
        except Exception:
            pass
    return parse_response(text, expected_count=len(crops))


@dataclass
class JudgeStats:
    total: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    ad_count: int = 0
    editorial_count: int = 0
    furniture_count: int = 0
    errors: int = 0


def judge_candidates(
    crops: List[bytes],
    page_context: Optional[str] = None,
    cache_model=None,
    db_session=None,
    max_api_calls: Optional[int] = None,
    stats: Optional[JudgeStats] = None,
) -> List[Verdict]:
    """Classify a list of cropped PNG byte blobs.

    Returns a Verdict per input crop, in the same order. Uses SHA256 cache
    lookups before calling Claude. Groups uncached crops into batches of up
    to MAX_IMAGES_PER_CALL per API request.

    Mock mode: if AD_JUDGE_MOCK=1 in environment, all crops return VERDICT_AD
    without calling the network. Useful for offline smoke testing.
    """
    n = len(crops)
    if stats is None:
        stats = JudgeStats()
    results: List[Optional[Verdict]] = [None] * n
    model = resolve_judge_model()

    if os.environ.get("AD_JUDGE_MOCK") == "1":
        for i, crop in enumerate(crops):
            v = _mock_verdict(crop)
            results[i] = v
            stats.total += 1
            _bump(stats, v.verdict)
        return [r for r in results if r is not None]  # type: ignore[misc]

    # 1) cache lookups
    hashes: List[str] = [_hash_crop(c, model) for c in crops]
    uncached_idx: List[int] = []
    for i, h in enumerate(hashes):
        cached = _cached_verdict(h, cache_model, db_session)
        if cached:
            results[i] = cached
            stats.cache_hits += 1
            stats.total += 1
            _bump(stats, cached.verdict)
        else:
            uncached_idx.append(i)

    if not uncached_idx:
        return [r for r in results if r is not None]  # type: ignore[misc]

    # 2) batch uncached crops
    api_budget = max_api_calls if max_api_calls is not None else DEFAULT_MAX_CALLS_PER_PAPER
    batches: List[List[int]] = []
    for start in range(0, len(uncached_idx), MAX_IMAGES_PER_CALL):
        batches.append(uncached_idx[start:start + MAX_IMAGES_PER_CALL])

    for batch in batches:
        if stats.api_calls >= api_budget:
            # Over budget: stamp remaining as EDITORIAL (drop safely)
            for i in batch:
                v = Verdict(
                    verdict=VERDICT_EDITORIAL,
                    reason="budget exceeded; defaulted to EDITORIAL",
                    model_used=f"budget-fallback ({model})",
                    cache_hit=False,
                    crop_hash=hashes[i],
                )
                results[i] = v
                stats.total += 1
                _bump(stats, v.verdict)
            continue

        batch_crops = [crops[i] for i in batch]
        try:
            pairs = _call_claude(batch_crops, page_context=page_context, model=model)
            stats.api_calls += 1
        except JudgeUnavailable:
            # propagate — caller decides whether to persist raw candidates unjudged
            raise
        except Exception as e:
            print(f"[ad_judge] API call failed: {e}")
            stats.errors += 1
            # OVER-MARK on judge outage: when the API can't classify (e.g. a
            # sustained 529 overload that outlasts the retries), KEEP the
            # candidates as ads rather than dropping them. A review-and-delete
            # paper beats a near-empty one, and matches the over-mark workflow.
            # These verdicts are deliberately NOT cached, so a later healthy
            # re-run will properly judge (and box-refine) the same crops.
            for i in batch:
                v = Verdict(
                    verdict=VERDICT_AD,
                    reason=f"judge unavailable (API error); kept unjudged for review: {e}",
                    model_used=f"unjudged-fallback ({model})",
                    cache_hit=False,
                    crop_hash=hashes[i],
                )
                results[i] = v
                stats.total += 1
                _bump(stats, v.verdict)
            continue

        for pos, i in enumerate(batch):
            verdict_str, reason, box = pairs[pos]
            v = Verdict(
                verdict=verdict_str,
                reason=reason,
                model_used=model,
                cache_hit=False,
                crop_hash=hashes[i],
                crop_box=box,
            )
            results[i] = v
            stats.total += 1
            _bump(stats, v.verdict)
            _persist_verdict(v, cache_model, db_session)

    return [r for r in results if r is not None]  # type: ignore[misc]


def _bump(stats: JudgeStats, verdict: str) -> None:
    if verdict == VERDICT_AD:
        stats.ad_count += 1
    elif verdict == VERDICT_EDITORIAL:
        stats.editorial_count += 1
    elif verdict == VERDICT_FURNITURE:
        stats.furniture_count += 1
