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
PROMPT_VERSION = "v2_2026-04-20_obits_plus_notices"

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
    "  and LEGAL / PUBLIC NOTICES (paid by government entities or law firms).\n"
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
    "- Legal notices, public hearing notices, and probate notices = AD. They often\n"
    "  include formal language like 'NOTICE IS HEREBY GIVEN', statute references,\n"
    "  or government body names.\n"
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
    "\n"
    "For each image in order, respond in this exact format, one stanza per image,\n"
    "stanzas separated by a blank line:\n"
    "IMAGE <n>\n"
    "VERDICT: AD|EDITORIAL|FURNITURE\n"
    "REASON: one short sentence.\n"
)


@dataclass
class Verdict:
    verdict: str            # "AD" | "EDITORIAL" | "FURNITURE"
    reason: str
    model_used: str
    cache_hit: bool
    crop_hash: str

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
    return Verdict(
        verdict=row.verdict,
        reason=row.reason or "",
        model_used=row.model_used or resolve_judge_model(),
        cache_hit=True,
        crop_hash=crop_hash,
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
            reason=v.reason[:500] if v.reason else None,
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


# IMAGE N stanza matcher. Tolerates extra whitespace and optional punctuation.
_STANZA_RE = re.compile(
    r"IMAGE\s*(\d+).*?VERDICT:\s*(AD|EDITORIAL|FURNITURE).*?REASON:\s*([^\n]+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_response(text: str, expected_count: int) -> List[tuple[str, str]]:
    """Return a list of (verdict, reason) tuples of length expected_count.

    Parser is permissive: if the model returns fewer stanzas than expected,
    missing entries default to EDITORIAL (safer: we'd rather drop an ad than
    persist an editorial false positive). If it returns more, we truncate.
    """
    out: List[tuple[str, str]] = [(VERDICT_EDITORIAL, "unparsed: no stanza found")] * expected_count
    for m in _STANZA_RE.finditer(text):
        try:
            idx = int(m.group(1)) - 1
            verdict = m.group(2).upper()
            reason = m.group(3).strip()
            if 0 <= idx < expected_count and verdict in _VALID_VERDICTS:
                out[idx] = (verdict, reason)
        except Exception:
            continue
    return out


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


def _call_claude(crops: List[bytes], page_context: Optional[str], model: str) -> List[tuple[str, str]]:
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
            # stamp this batch as EDITORIAL (safer: drop unknown)
            for i in batch:
                v = Verdict(
                    verdict=VERDICT_EDITORIAL,
                    reason=f"API error: {e}",
                    model_used=model,
                    cache_hit=False,
                    crop_hash=hashes[i],
                )
                results[i] = v
                stats.total += 1
                _bump(stats, v.verdict)
            continue

        for pos, i in enumerate(batch):
            verdict_str, reason = pairs[pos]
            v = Verdict(
                verdict=verdict_str,
                reason=reason,
                model_used=model,
                cache_hit=False,
                crop_hash=hashes[i],
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
