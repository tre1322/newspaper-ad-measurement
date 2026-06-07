"""Measure the REAL detector against human marks by running it blind.

Why a 'shadow' publication
--------------------------
The production pipeline (_detect_and_save_ads) seeds its candidate list with the
page's existing AdBoxes and dedupes against them — so running it on a paper that
already holds the human marks makes it skip exactly the regions we want it to
find on its own. That's answer-key leakage.

So we build a SHADOW publication from the same source PDF, render only the
page(s) that have ground-truth marks, and run the unmodified production detector
on it (no seeded boxes — blind). Then we score the shadow's auto boxes against
the original publication's human marks with the tested area-coverage scorer in
ground_truth_eval.py. Same PDF rendered at the same 1.5x zoom => identical pixel
coordinate space, so the two box sets are directly comparable.

This uses the actual production code path end-to-end, so the number reflects
what a real upload of this paper would produce today.

    venv\\Scripts\\python.exe rerun_eval.py --gt-pub 26 --pdf OA-2025-01-01.pdf
    venv\\Scripts\\python.exe rerun_eval.py --gt-pub 26 --pages 6 --keep
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import fitz

from app import (app, db, Publication, Page, AdBox, PUBLICATION_CONFIGS,
                 _detect_and_save_ads)
from ground_truth_eval import page_area_scores, gt_boxes_found, _pct

ZOOM = 1.5  # must match the production render (fitz.Matrix(1.5, 1.5)) so the
            # shadow's pixel coordinates line up with the stored human marks.


def _gt_pages(gt_pub_id, only_pages):
    """page_numbers in the GT pub that actually carry human marks."""
    rows = (db.session.query(Page.page_number)
            .join(AdBox, AdBox.page_id == Page.id)
            .filter(Page.publication_id == gt_pub_id,
                    AdBox.detected_automatically == False)  # noqa: E712
            .distinct().all())
    pages = sorted(r[0] for r in rows)
    if only_pages:
        pages = [p for p in pages if p in only_pages]
    return pages


def _gt_boxes(gt_pub_id, page_number):
    rows = (db.session.query(AdBox)
            .join(Page, AdBox.page_id == Page.id)
            .filter(Page.publication_id == gt_pub_id,
                    Page.page_number == page_number,
                    AdBox.detected_automatically == False)  # noqa: E712
            .all())
    return [(b.x, b.y, b.width, b.height) for b in rows]


def build_shadow(src_pdf, gt_pub, pages, upload_folder):
    """Stage the PDF + render the GT pages into a fresh, box-free publication."""
    pdf_dir = os.path.join(upload_folder, "pdfs")
    png_dir = os.path.join(upload_folder, "pages")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    staged_name = f"shadow_{uuid.uuid4().hex[:8]}.pdf"
    shutil.copyfile(src_pdf, os.path.join(pdf_dir, staged_name))

    cfg = PUBLICATION_CONFIGS.get(gt_pub.publication_type,
                                  PUBLICATION_CONFIGS["broadsheet"])
    shadow = Publication(
        filename=staged_name,
        # original_filename drives the masthead ('oa') for shared-header parity,
        # so a fresh-upload of this paper would behave the same way.
        original_filename=gt_pub.original_filename,
        publication_type=gt_pub.publication_type,
        total_pages=gt_pub.total_pages,
        total_inches=cfg["total_inches_per_page"] * gt_pub.total_pages,
        processed=False,
    )
    db.session.add(shadow)
    db.session.commit()

    doc = fitz.open(src_pdf)
    for pnum in pages:
        page = doc[pnum - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
        pix.save(os.path.join(png_dir, f"{staged_name}_page_{pnum}.png"))
        db.session.add(Page(
            publication_id=shadow.id, page_number=pnum,
            width_pixels=pix.width, height_pixels=pix.height,
            total_page_inches=cfg["total_inches_per_page"],
        ))
    doc.close()
    db.session.commit()
    return shadow, staged_name


def cleanup_shadow(shadow_id, staged_name, upload_folder):
    pages = Page.query.filter_by(publication_id=shadow_id).all()
    for pg in pages:
        AdBox.query.filter_by(page_id=pg.id).delete()
    for pg in pages:
        db.session.delete(pg)
    pub = db.session.get(Publication, shadow_id)
    if pub:
        db.session.delete(pub)
    db.session.commit()
    # files
    try:
        os.remove(os.path.join(upload_folder, "pdfs", staged_name))
    except OSError:
        pass
    for pg in pages:
        try:
            os.remove(os.path.join(upload_folder, "pages",
                                   f"{staged_name}_page_{pg.page_number}.png"))
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Run the real detector blind and score it vs human marks.")
    ap.add_argument("--gt-pub", type=int, default=26, help="publication id holding the human marks")
    ap.add_argument("--pdf", default="OA-2025-01-01.pdf", help="source PDF (same paper as --gt-pub)")
    ap.add_argument("--pages", type=int, nargs="*", help="limit to these page numbers (default: all marked pages)")
    ap.add_argument("--found-thresh", type=float, default=0.5)
    ap.add_argument("--keep", action="store_true", help="do not delete the shadow publication afterward")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Source PDF not found: {args.pdf}")
        sys.exit(1)

    with app.app_context():
        upload_folder = app.config["UPLOAD_FOLDER"]
        gt_pub = db.session.get(Publication, args.gt_pub)
        if not gt_pub:
            print(f"GT publication {args.gt_pub} not found")
            sys.exit(1)

        pages = _gt_pages(args.gt_pub, set(args.pages) if args.pages else None)
        if not pages:
            print(f"GT pub {args.gt_pub} has no human-marked pages "
                  f"{'in '+str(args.pages) if args.pages else ''}. Nothing to score.")
            sys.exit(1)

        print(f"GT pub {args.gt_pub} ({gt_pub.original_filename}) — marked pages: {pages}")
        print(f"Building blind shadow from {args.pdf} (pages {pages})...\n")
        shadow, staged_name = build_shadow(args.pdf, gt_pub, pages, upload_folder)

        try:
            result = _detect_and_save_ads(shadow.id)
            print(f"\n[rerun] detector saved {result.get('saved')} boxes; "
                  f"judge stats: {result.get('judge_stats')}\n")

            tot = {"inter": 0, "union": 0, "gt_area": 0, "pred_area": 0}
            f_sum = t_sum = 0
            hdr = f"{'page':>4}  {'recall':>7} {'prec':>7} {'iou':>7}  {'ads':>9}  preds"
            print(hdr); print("-" * len(hdr))
            for pnum in pages:
                spage = Page.query.filter_by(publication_id=shadow.id, page_number=pnum).first()
                gt = _gt_boxes(args.gt_pub, pnum)
                pred = [(b.x, b.y, b.width, b.height)
                        for b in AdBox.query.filter_by(page_id=spage.id).all()]
                W, H = spage.width_pixels, spage.height_pixels
                s = page_area_scores(gt, pred, W, H)
                fnd, tot_ads, missed = gt_boxes_found(gt, pred, W, H, args.found_thresh)
                for k in tot:
                    tot[k] += s[k]
                f_sum += fnd; t_sum += tot_ads
                print(f"{pnum:>4}  {_pct(s['recall'])} {_pct(s['precision'])} "
                      f"{_pct(s['iou'])}  {f'{fnd}/{tot_ads}':>9}  {len(pred)}")
                for m in missed:
                    print(f"        MISSED ad at {m['box']} (only {m['covered']*100:.0f}% covered)")

            o_r = tot["inter"] / tot["gt_area"] if tot["gt_area"] else None
            o_p = tot["inter"] / tot["pred_area"] if tot["pred_area"] else None
            o_i = tot["inter"] / tot["union"] if tot["union"] else None
            print("-" * len(hdr))
            print(f"{'ALL':>4}  {_pct(o_r)} {_pct(o_p)} {_pct(o_i)}  {f'{f_sum}/{t_sum}':>9}")
            print(f"\nRECALL    {_pct(o_r)}  — share of your marked ad AREA the detector found")
            print(f"PRECISION {_pct(o_p)}  — share of detector-marked AREA that was really ad")
            print(f"ADS       {f_sum}/{t_sum} of your individual marked ads ≥{int(args.found_thresh*100)}% covered")
        finally:
            if args.keep:
                print(f"\n[rerun] keeping shadow pub id={shadow.id} (filename {staged_name})")
            else:
                cleanup_shadow(shadow.id, staged_name, upload_folder)
                print(f"\n[rerun] cleaned up shadow pub id={shadow.id}")


if __name__ == "__main__":
    main()
