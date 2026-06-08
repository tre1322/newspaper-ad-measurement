"""Score the ad-detector against human-marked ground truth, by ad-AREA coverage.

You have hundreds of papers whose ads were marked by hand. Those manual marks
are gold-standard labels. This harness turns "the detector feels ~65% accurate"
into a real, reproducible number — per page, per publication, and overall —
using the metric Trevor cares about: ad-AREA coverage, NOT box-count match.

WHY AREA, NOT BOX COUNT
-----------------------
Human markup granularity is inconsistent (sometimes one box around 15 sponsor
tiles, sometimes 15 separate boxes). Box-count matching punishes the detector
for a granularity difference that doesn't matter. So we rasterize every box onto
a full-resolution page mask and compare the UNION of ground-truth ad pixels to
the UNION of predicted ad pixels. One box or fifteen over the same region paint
the same pixels, so the score is identical either way.

  recall    = (GT ∩ PRED) / GT      "what fraction of real ad area did we catch?"
  precision = (GT ∩ PRED) / PRED    "what fraction of marked area was really ad?"
  iou       = (GT ∩ PRED) / (GT ∪ PRED)

GROUND TRUTH vs PREDICTION
--------------------------
In this schema, ad_box only ever stores confirmed ADS (is_ad is always 1).
  * Ground truth  = boxes a human drew     -> detected_automatically = 0
  * Prediction    = boxes the detector left -> detected_automatically = 1  (mode 'auto')
                    or a fresh pipeline run                                  (mode 'rerun')

A publication a human reviewed is scored on ALL its pages: a page with no human
boxes is ground-truth "no ads here", so any predicted box there counts against
precision. That correctly penalizes false positives on editorial pages.

USAGE
-----
    venv\\Scripts\\python.exe ground_truth_eval.py --selftest      # verify the scorer (no DB)
    venv\\Scripts\\python.exe ground_truth_eval.py                 # score every GT pub (stored auto preds)
    venv\\Scripts\\python.exe ground_truth_eval.py --pub 60        # one publication
    venv\\Scripts\\python.exe ground_truth_eval.py --json out.json # also write a machine-readable report

Point it at prod by setting DATABASE_URL before running (read-only SELECTs):
    $env:DATABASE_URL = "<railway postgres url>"; venv\\Scripts\\python.exe ground_truth_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# The pure scorer lives in area_score.py so the live app can import it without
# pulling in this CLI script. Re-exported here for the harness's callers.
from area_score import _mask_for, page_area_scores, gt_boxes_found  # noqa: F401


def _pct(v):
    return "  n/a" if v is None else f"{v * 100:5.1f}%"


# ---------------------------------------------------------------------------
# Self-test: analytic cases with hand-computed answers. Run with --selftest.
# ---------------------------------------------------------------------------

def selftest():
    def approx(a, b, tol=1e-9):
        return a is not None and abs(a - b) < tol

    # 1) identical boxes -> perfect
    s = page_area_scores([(0, 0, 100, 100)], [(0, 0, 100, 100)], 200, 200)
    assert approx(s["recall"], 1) and approx(s["precision"], 1) and approx(s["iou"], 1), s

    # 2) half-offset overlap: inter=50*50=2500, union=2*1e4-2500=17500
    s = page_area_scores([(0, 0, 100, 100)], [(50, 50, 100, 100)], 200, 200)
    assert approx(s["recall"], 0.25) and approx(s["precision"], 0.25), s
    assert approx(s["iou"], 2500 / 17500), s

    # 3) disjoint -> zeros
    s = page_area_scores([(0, 0, 50, 50)], [(100, 100, 50, 50)], 200, 200)
    assert s["recall"] == 0 and s["precision"] == 0 and s["iou"] == 0, s

    # 4) GRANULARITY: two overlapping GT boxes (union 150x100) vs one pred box
    #    covering the same span -> perfect. Proves box-count doesn't matter.
    s = page_area_scores([(0, 0, 100, 100), (50, 0, 100, 100)],
                         [(0, 0, 150, 100)], 200, 200)
    assert approx(s["recall"], 1) and approx(s["precision"], 1) and approx(s["iou"], 1), s

    # 5) false positive on an ad-free page: gt empty, pred non-empty
    s = page_area_scores([], [(0, 0, 100, 100)], 200, 200)
    assert s["recall"] is None and s["precision"] == 0 and s["iou"] == 0, s

    # 6) per-box found: pred covers box A fully, box B not at all
    found, total, missed = gt_boxes_found(
        [(0, 0, 100, 100), (300, 300, 100, 100)], [(0, 0, 100, 100)], 500, 500)
    assert found == 1 and total == 2 and len(missed) == 1, (found, total, missed)

    print("selftest: all assertions passed ✓")


# ---------------------------------------------------------------------------
# DB-backed evaluation.
# ---------------------------------------------------------------------------

def run_eval(pub_ids=None, pred_mode="auto", limit=None, json_path=None, found_thresh=0.5):
    from app import app, db, Publication, Page, AdBox

    with app.app_context():
        # GT pubs = publications that have at least one human-drawn box.
        gt_pub_q = (db.session.query(Publication.id)
                    .join(Page, Page.publication_id == Publication.id)
                    .join(AdBox, AdBox.page_id == Page.id)
                    .filter(AdBox.detected_automatically == False)  # noqa: E712
                    .distinct())
        if pub_ids:
            gt_pub_q = gt_pub_q.filter(Publication.id.in_(pub_ids))
        ids = sorted(r[0] for r in gt_pub_q.all())
        if limit:
            ids = ids[:limit]

        if not ids:
            print("No publications with human-marked boxes found "
                  "(detected_automatically=0). Nothing to score.")
            return

        # Micro (area-weighted) accumulators across every scored page.
        tot = {"inter": 0, "union": 0, "gt_area": 0, "pred_area": 0}
        found_sum = total_sum = 0
        pub_reports = []

        print(f"Scoring {len(ids)} publication(s) | prediction source = {pred_mode!r}\n")
        header = f"{'pub':>4}  {'file':<22} {'pg':>3}  {'recall':>7} {'prec':>7} {'iou':>7}  {'ads':>9}"
        print(header)
        print("-" * len(header))

        for pid in ids:
            pub = db.session.get(Publication, pid)
            pages = (Page.query.filter_by(publication_id=pid)
                     .order_by(Page.page_number).all())
            p_tot = {"inter": 0, "union": 0, "gt_area": 0, "pred_area": 0}
            p_found = p_total = 0
            page_rows = []

            for page in pages:
                boxes = AdBox.query.filter_by(page_id=page.id).all()
                gt = [(b.x, b.y, b.width, b.height) for b in boxes
                      if not b.detected_automatically]
                if pred_mode == "auto":
                    pred = [(b.x, b.y, b.width, b.height) for b in boxes
                            if b.detected_automatically]
                elif pred_mode == "rerun":
                    raise NotImplementedError(
                        "rerun mode regenerates predictions from the source PDF and "
                        "is wired separately (needs PDFs on disk + costs API). Use "
                        "the default 'auto' mode against a DB that has detector output.")
                else:
                    raise ValueError(f"unknown pred_mode {pred_mode!r}")

                W = int(page.width_pixels or 0)
                H = int(page.height_pixels or 0)
                if W <= 0 or H <= 0:
                    continue
                s = page_area_scores(gt, pred, W, H)
                f, t, missed = gt_boxes_found(gt, pred, W, H, found_thresh)
                for k in p_tot:
                    p_tot[k] += s[k]
                p_found += f
                p_total += t
                page_rows.append({"page": page.page_number, **s,
                                  "found": f, "total": t, "missed": missed})

            # publication-level micro scores
            p_recall = p_tot["inter"] / p_tot["gt_area"] if p_tot["gt_area"] else None
            p_prec = p_tot["inter"] / p_tot["pred_area"] if p_tot["pred_area"] else None
            p_iou = p_tot["inter"] / p_tot["union"] if p_tot["union"] else None
            ads_str = f"{p_found}/{p_total}"
            print(f"{pid:>4}  {(pub.original_filename or '')[:22]:<22} "
                  f"{len(pages):>3}  {_pct(p_recall)} {_pct(p_prec)} {_pct(p_iou)}  {ads_str:>9}")

            for k in tot:
                tot[k] += p_tot[k]
            found_sum += p_found
            total_sum += p_total
            pub_reports.append({
                "pub_id": pid, "file": pub.original_filename, "pages": len(pages),
                "recall": p_recall, "precision": p_prec, "iou": p_iou,
                "ads_found": p_found, "ads_total": p_total, "page_detail": page_rows,
            })

        # overall micro (area-weighted) headline
        o_recall = tot["inter"] / tot["gt_area"] if tot["gt_area"] else None
        o_prec = tot["inter"] / tot["pred_area"] if tot["pred_area"] else None
        o_iou = tot["inter"] / tot["union"] if tot["union"] else None
        print("-" * len(header))
        print(f"{'ALL':>4}  {'(area-weighted)':<22} {'':>3}  "
              f"{_pct(o_recall)} {_pct(o_prec)} {_pct(o_iou)}  {found_sum}/{total_sum}")
        print(f"\nRecall   {_pct(o_recall)}  — share of human-marked ad AREA the detector caught")
        print(f"Precision{_pct(o_prec)}  — share of detector-marked AREA that was really ad")
        print(f"Ads      {found_sum}/{total_sum} individual GT ad boxes ≥{int(found_thresh*100)}% covered")

        if json_path:
            d = os.path.dirname(json_path)
            if d:
                os.makedirs(d, exist_ok=True)
            report = {
                "pred_mode": pred_mode, "found_thresh": found_thresh,
                "overall": {"recall": o_recall, "precision": o_prec, "iou": o_iou,
                            "ads_found": found_sum, "ads_total": total_sum, **tot},
                "publications": pub_reports,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nWrote {json_path}")


def main():
    ap = argparse.ArgumentParser(description="Score the ad-detector vs human ground truth (ad-area coverage).")
    ap.add_argument("--selftest", action="store_true", help="run analytic scorer checks (no DB) and exit")
    ap.add_argument("--pub", type=int, nargs="*", help="only score these publication id(s)")
    ap.add_argument("--pred", default="auto", choices=["auto", "rerun"], help="prediction source")
    ap.add_argument("--limit", type=int, help="cap number of publications scored")
    ap.add_argument("--found-thresh", type=float, default=0.5, help="GT box 'found' if this fraction of its area is covered")
    ap.add_argument("--json", dest="json_path", help="write a machine-readable report here")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    run_eval(pub_ids=args.pub, pred_mode=args.pred, limit=args.limit,
             json_path=args.json_path, found_thresh=args.found_thresh)


if __name__ == "__main__":
    main()
