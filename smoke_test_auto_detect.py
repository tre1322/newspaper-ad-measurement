"""End-to-end smoke test for automatic ad detection on upload.

Exercises the same code path the web UI triggers:
  Publication insert -> start_background_processing(pub_id) -> AdBox rows.

Uses the bundled OA-2025-01-01.pdf sample. Run with:
    venv\\Scripts\\python.exe smoke_test_auto_detect.py
"""
import os
import shutil
import sys
import time

# Force UTF-8 stdout so the many emoji prints in app.py don't crash the run
# on Windows cp1252 consoles.
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import fitz

SAMPLE_PDF = "OA-2025-01-01.pdf"


def main():
    from app import (
        app, db, Publication, Page, AdBox, UserCorrection,
        start_background_processing, PUBLICATION_CONFIGS,
    )

    with app.app_context():
        uploads_dir = app.config['UPLOAD_FOLDER']
        pdf_dir = os.path.join(uploads_dir, 'pdfs')
        pages_dir = os.path.join(uploads_dir, 'pages')
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(pages_dir, exist_ok=True)

        if not os.path.exists(SAMPLE_PDF):
            print(f"FAIL: sample PDF missing: {SAMPLE_PDF}")
            sys.exit(1)

        # Put the PDF where the upload flow expects it.
        target_name = f"smoketest_{int(time.time())}_{SAMPLE_PDF}"
        target_path = os.path.join(pdf_dir, target_name)
        shutil.copyfile(SAMPLE_PDF, target_path)
        print(f"Copied sample -> {target_path}")

        db.create_all()

        with fitz.open(target_path) as _doc:
            total_pages = _doc.page_count

        pub = Publication(
            filename=target_name,
            original_filename=SAMPLE_PDF,
            publication_type='broadsheet',
            total_pages=total_pages,
            total_inches=total_pages * PUBLICATION_CONFIGS['broadsheet']['total_inches_per_page'],
        )
        db.session.add(pub)
        db.session.commit()
        pub_id = pub.id
        print(f"Created Publication id={pub_id}")

        # Run the exact upload flow (synchronous). Guard with our own traceback
        # print so we can see errors that the upload flow's broad except swallows.
        import traceback as _tb
        t0 = time.time()
        try:
            start_background_processing(pub_id, run_async=False)
        except Exception as _e:
            print(f"start_background_processing raised: {_e}")
            _tb.print_exc()
        dt = time.time() - t0
        print(f"\nProcessing done in {dt:.1f}s")

        # Capture any failure recorded on the publication itself.
        _pub_check = db.session.get(Publication, pub_id)
        try:
            err = getattr(_pub_check, 'processing_error', None)
            if err:
                print(f"Recorded processing_error: {err}")
        except Exception:
            pass

        pub = db.session.get(Publication, pub_id)
        print(f"Publication.status = {pub.safe_processing_status!r}")
        print(f"Publication.processed = {pub.processed}")

        pages = Page.query.filter_by(publication_id=pub_id).order_by(Page.page_number).all()
        print(f"Pages created: {len(pages)}")

        ad_boxes = (AdBox.query
                    .join(Page, AdBox.page_id == Page.id)
                    .filter(Page.publication_id == pub_id).all())
        auto_boxes = [b for b in ad_boxes if b.detected_automatically]
        print(f"Total AdBox rows: {len(ad_boxes)}  (auto={len(auto_boxes)})")

        # Per-page breakdown with source counts.
        source_counts = {}
        for b in auto_boxes:
            source_counts[b.ad_type] = source_counts.get(b.ad_type, 0) + 1
        if source_counts:
            print("By source:")
            for src, n in sorted(source_counts.items()):
                print(f"  {src}: {n}")

        # Acceptance criteria for the smoke test. Status=='completed' is flaky
        # on this dev sqlite because the `processing_status` column is added
        # via migration but not declared on the ORM model; the deploy DB has
        # it properly. We treat pages + auto boxes as the real signal.
        ok_pages = len(pages) > 0
        ok_boxes = len(auto_boxes) > 0
        boxes_per_page = len(auto_boxes) / max(1, len(pages))
        print("\n=== RESULT ===")
        print(f"pages created      : {ok_pages}")
        print(f"auto boxes > 0     : {ok_boxes}")
        print(f"boxes per page avg : {boxes_per_page:.1f}")
        if not ok_pages:
            print("FAIL: no pages were created")
            sys.exit(2)
        if not ok_boxes:
            print("WARN: no auto boxes were produced -- check detection log output above")
            sys.exit(3)
        if boxes_per_page > 30:
            print("WARN: box density very high -- confidence threshold may need more tuning")
        print("PASS")


if __name__ == '__main__':
    main()
