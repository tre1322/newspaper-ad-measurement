"""READ-ONLY scoping scan of prod ground truth. No writes, no app import.

Answers: how many prod papers are hand-marked, how complete the markups are,
per-masthead spread, and — where a paper already has detector output beside the
marks — a free area-coverage score. Connects with raw SQLAlchemy SELECTs so it
never triggers app.py's import-time db.create_all / ALTER TABLE on prod.

DATABASE_URL is read from env or the gitignored .railway_vars.json; never printed.
"""
import json
import os
import re

from sqlalchemy import create_engine, text

from ground_truth_eval import page_area_scores  # pure numpy, no app import


def _db_url():
    u = os.environ.get("DATABASE_URL")
    if u:
        return u
    with open(".railway_vars.json", encoding="utf-8-sig") as f:
        return json.load(f)["DATABASE_URL"]


def _masthead(name):
    m = re.match(r"[a-z]+", (name or "").strip().lower())
    return m.group(0) if m else "?"


def main():
    engine = create_engine(_db_url(), connect_args={"connect_timeout": 20})
    with engine.connect() as c:
        # 1) headline counts
        pubs = c.execute(text("SELECT count(*) FROM publication")).scalar()
        pages = c.execute(text("SELECT count(*) FROM page")).scalar()
        boxes = c.execute(text(
            "SELECT detected_automatically, count(*) FROM ad_box GROUP BY 1")).all()
        print(f"publications: {pubs}   pages: {pages}")
        print("ad_box by detected_automatically:",
              {('auto' if k else 'manual'): v for k, v in boxes})

        # 2) marked / auto / both
        marked = c.execute(text("""
            SELECT count(DISTINCT p.id) FROM publication p
            JOIN page pg ON pg.publication_id=p.id
            JOIN ad_box b ON b.page_id=pg.id
            WHERE b.detected_automatically = false""")).scalar()
        has_auto = c.execute(text("""
            SELECT count(DISTINCT p.id) FROM publication p
            JOIN page pg ON pg.publication_id=p.id
            JOIN ad_box b ON b.page_id=pg.id
            WHERE b.detected_automatically = true""")).scalar()
        print(f"\npubs with HUMAN marks: {marked}   pubs with AUTO boxes: {has_auto}")

        # 3) per-publication completeness + masthead (only pubs that have marks)
        rows = c.execute(text("""
            SELECT p.id, p.original_filename,
                   count(DISTINCT pg.id) AS npages,
                   count(DISTINCT CASE WHEN b.detected_automatically=false THEN pg.id END) AS marked_pages,
                   count(CASE WHEN b.detected_automatically=false THEN 1 END) AS man_boxes,
                   count(CASE WHEN b.detected_automatically=true  THEN 1 END) AS auto_boxes
            FROM publication p
            JOIN page pg ON pg.publication_id=p.id
            LEFT JOIN ad_box b ON b.page_id=pg.id
            GROUP BY p.id, p.original_filename
            HAVING count(CASE WHEN b.detected_automatically=false THEN 1 END) > 0
            ORDER BY p.id""")).all()

        by_mast = {}
        full = partial = 0
        both_pubs = []
        for pid, fn, npg, mpg, man, auto in rows:
            mast = _masthead(fn)
            d = by_mast.setdefault(mast, {"pubs": 0, "man_boxes": 0})
            d["pubs"] += 1
            d["man_boxes"] += man or 0
            frac = (mpg / npg) if npg else 0
            if frac >= 0.8:
                full += 1
            else:
                partial += 1
            if auto and man:
                both_pubs.append(pid)

        print(f"\nmarked pubs: {len(rows)}  | markup completeness: "
              f"{full} cover >=80% of pages, {partial} are partial")
        print("by masthead:")
        for mast, d in sorted(by_mast.items(), key=lambda kv: -kv[1]["pubs"]):
            print(f"  {mast:<6} {d['pubs']:>4} marked pubs   {d['man_boxes']:>5} hand-marked ads")

        # 4) free score where auto output already sits beside the marks
        print(f"\npubs with BOTH human + auto boxes (free-scoreable): {len(both_pubs)}")
        if both_pubs:
            tot = {"inter": 0, "union": 0, "gt_area": 0, "pred_area": 0}
            for pid in both_pubs:
                prows = c.execute(text("""
                    SELECT pg.id, pg.width_pixels, pg.height_pixels FROM page pg
                    WHERE pg.publication_id=:pid"""), {"pid": pid}).all()
                for page_id, W, H in prows:
                    if not W or not H:
                        continue
                    brows = c.execute(text("""
                        SELECT x,y,width,height,detected_automatically
                        FROM ad_box WHERE page_id=:p"""), {"p": page_id}).all()
                    gt = [(r[0], r[1], r[2], r[3]) for r in brows if not r[4]]
                    pred = [(r[0], r[1], r[2], r[3]) for r in brows if r[4]]
                    if not gt and not pred:
                        continue
                    s = page_area_scores(gt, pred, int(W), int(H))
                    for k in tot:
                        tot[k] += s[k]
            r = tot["inter"] / tot["gt_area"] if tot["gt_area"] else None
            pr = tot["inter"] / tot["pred_area"] if tot["pred_area"] else None
            fmt = lambda v: "n/a" if v is None else f"{v*100:.1f}%"
            print(f"  area-weighted over those pubs: recall {fmt(r)}  precision {fmt(pr)}")
            print("  (caveat: 'auto' here is post-review output, not a blind run — "
                  "indicative only; the real number needs rerun mode)")


if __name__ == "__main__":
    main()
