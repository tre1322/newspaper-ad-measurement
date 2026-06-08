"""End-to-end smoke for the detector-accuracy wiring.

Exercises the real user path: detector output -> snapshot -> human review
(delete a false positive, add a missed ad) -> GET /api/detector_accuracy.
Uses known geometry so the scores are exactly predictable.

    venv\\Scripts\\python.exe smoke_detector_accuracy.py
"""
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app import (app, db, Publication, Page, AdBox, UserCorrection,
                 DetectorSnapshot, _snapshot_detector_output)


def _box(page, x, y, w, h, auto):
    return AdBox(page_id=page.id, x=x, y=y, width=w, height=h,
                width_inches_raw=0.0, height_inches_raw=0.0,
                width_inches_rounded=0.0, height_inches_rounded=0.0,
                column_inches=0.0, ad_type='auto' if auto else 'manual',
                is_ad=True, detected_automatically=auto)


def main():
    with app.app_context():
        db.create_all()  # ensure detector_snapshot table exists
        pub = Publication(filename='smoke_acc.pdf', original_filename='smoke_acc.pdf',
                          publication_type='broadsheet', total_pages=1, total_inches=0.0,
                          processed=True)
        db.session.add(pub); db.session.commit()
        page = Page(publication_id=pub.id, page_number=1, width_pixels=1332,
                    height_pixels=2340, total_page_inches=0.0)
        db.session.add(page); db.session.commit()

        # Detector output: A (true positive) + B (false positive).
        A = _box(page, 100, 100, 200, 200, auto=True)
        B = _box(page, 500, 500, 200, 200, auto=True)
        db.session.add_all([A, B]); db.session.commit()

        # Freeze the prediction (captures A and B).
        _snapshot_detector_output(pub.id)
        snap = DetectorSnapshot.query.filter_by(publication_id=pub.id).first()
        assert snap is not None and snap.box_count == 2, ('snapshot', snap)

        # Human review: delete B (false positive) + log the deletion; add C (a miss).
        db.session.delete(B)
        db.session.add(UserCorrection(publication_id=pub.id, page_id=page.id,
                       x=500, y=500, width=200, height=200, is_ad=False,
                       correction_type='deleted'))
        C = _box(page, 800, 800, 200, 200, auto=False)
        db.session.add(C); db.session.commit()

        # Score via the real HTTP endpoint.
        client = app.test_client()
        resp = client.get(f'/api/detector_accuracy/{pub.id}')
        assert resp.status_code == 200, resp.status_code
        data = resp.get_json()

        ok = True
        def check(name, got, want):
            nonlocal ok
            good = abs(got - want) < 1e-6 if isinstance(want, float) else got == want
            ok = ok and good
            print(f"  {'OK ' if good else 'BAD'} {name}: got {got} want {want}")

        print("response overall:", data['overall'])
        check('available', data['available'], True)
        check('reviewed', data['reviewed'], True)
        check('recall', data['overall']['recall'], 0.5)       # A∩truth / truth(A+C)
        check('precision', data['overall']['precision'], 0.5)  # A∩pred / pred(A+B)
        check('ads_found', data['overall']['ads_found'], 1)    # A found, C missed
        check('ads_total', data['overall']['ads_total'], 2)

        # Also confirm the not-found endpoint behaviour for a pub with no snapshot.
        resp2 = client.get('/api/detector_accuracy/99999999')
        check('missing_pub_404', resp2.status_code, 404)

        # Cleanup
        AdBox.query.filter_by(page_id=page.id).delete()
        UserCorrection.query.filter_by(publication_id=pub.id).delete()
        DetectorSnapshot.query.filter_by(publication_id=pub.id).delete()
        db.session.delete(page); db.session.delete(pub); db.session.commit()

        print("\nSMOKE PASS" if ok else "\nSMOKE FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
