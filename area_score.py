"""Pure ad-area-coverage scoring — no DB, no app, no images, just geometry.

Shared by the eval harness (ground_truth_eval.py, rerun_eval.py) and the live
app (app.py's /api/detector_accuracy). Kept dependency-free (numpy only) so the
production app can import it without pulling in the CLI eval scripts.

The metric rasterizes box sets onto a full-resolution page mask and compares the
UNION of ground-truth ad pixels to the UNION of predicted pixels, so box
granularity (one box vs fifteen over the same region) doesn't affect the score.
"""
from __future__ import annotations

import numpy as np


def _mask_for(boxes, W, H):
    """Paint boxes (x, y, w, h in page pixels) onto a boolean page mask (clipped)."""
    m = np.zeros((H, W), dtype=bool)
    for x, y, w, h in boxes:
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(W, int(round(x + w)))
        y1 = min(H, int(round(y + h)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def page_area_scores(gt_boxes, pred_boxes, W, H):
    """Area-coverage scores for one page. Ratios are None when undefined."""
    gt = _mask_for(gt_boxes, W, H)
    pr = _mask_for(pred_boxes, W, H)
    inter = int(np.logical_and(gt, pr).sum())
    union = int(np.logical_or(gt, pr).sum())
    gta = int(gt.sum())
    pra = int(pr.sum())
    return {
        "gt_area": gta,
        "pred_area": pra,
        "inter": inter,
        "union": union,
        "recall": (inter / gta) if gta else None,
        "precision": (inter / pra) if pra else None,
        "iou": (inter / union) if union else None,
    }


def gt_boxes_found(gt_boxes, pred_boxes, W, H, thresh=0.5):
    """How many individual GT boxes are >= thresh covered by the prediction mask.
    Returns (found_count, total_count, missed_boxes)."""
    pr = _mask_for(pred_boxes, W, H)
    found = 0
    missed = []
    for box in gt_boxes:
        x, y, w, h = box
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
        if x1 <= x0 or y1 <= y0:
            continue
        sub = pr[y0:y1, x0:x1]
        covered = sub.sum() / sub.size if sub.size else 0.0
        if covered >= thresh:
            found += 1
        else:
            missed.append({"box": [round(v, 1) for v in box], "covered": round(covered, 3)})
    return found, len(gt_boxes), missed
