"""Smoke test: measure-page button renders + save/list endpoints work (no API).

Run: venv\\Scripts\\python.exe smoke_recurring_template.py
"""
import os
os.environ.setdefault("AD_JUDGE_MOCK", "1")
from app import app, db, Publication, Page, AdBox, RecurringPageTemplate, _masthead

with app.app_context():
    # Find a page that already has at least one ad box (so save has something).
    pg = (db.session.query(Page).join(AdBox, AdBox.page_id == Page.id)
          .order_by(Page.id).first())
    assert pg, "no page with ad boxes in local DB"
    pub = db.session.get(Publication, pg.publication_id)
    pub_id, pnum, page_id = pub.id, pg.page_number, pg.id
    mh = _masthead(pub.original_filename)
    nbox = AdBox.query.filter_by(page_id=page_id).count()
    # ensure a clean slate for this masthead/label
    RecurringPageTemplate.query.filter_by(masthead=mh, label='smoketest').delete()
    db.session.commit()
print(f"using pub={pub_id} page={pnum} page_id={page_id} masthead={mh} boxes={nbox}")

c = app.test_client()
# 1. login then render the measure page -> button + JS must be present
c.post('/login', data={'password': 'CCCitizen56101!'}, follow_redirects=True)
r = c.get(f'/measure/{pub_id}/page/{pnum}')
html = r.get_data(as_text=True)
assert r.status_code == 200, f"measure page status {r.status_code}"
assert 'id="saveRecurringBtn"' in html, "button missing from rendered page"
assert 'function saveRecurringTemplate' in html, "JS handler missing"
assert '/api/save_recurring_template/' in html, "fetch URL missing"
print("ok: measure page renders 200 with the Save-as-Recurring button + handler")

# 2. POST the save endpoint -> records a template
r2 = c.post(f'/api/save_recurring_template/{page_id}',
            json={'label': 'smoketest'})
d2 = r2.get_json()
assert r2.status_code == 200 and d2.get('success'), f"save failed: {r2.status_code} {d2}"
assert d2['boxes'] == nbox, f"expected {nbox} boxes, got {d2['boxes']}"
print(f"ok: save endpoint recorded {d2['boxes']} boxes for masthead '{d2['masthead']}'")

# 3. GET the diagnostic list -> template appears
r3 = c.get('/api/recurring_templates')
d3 = r3.get_json()
found = [t for t in d3.get('templates', [])
         if t['masthead'] == mh and t['label'] == 'smoketest']
assert found and found[0]['box_count'] == nbox, f"template not listed: {d3}"
print(f"ok: diagnostic lists the template (anchors={found[0]['anchors'][:1]}...)")

# 4. empty-page guard: saving a page with no boxes must 400
with app.app_context():
    empty = (db.session.query(Page)
             .outerjoin(AdBox, AdBox.page_id == Page.id)
             .filter(AdBox.id.is_(None)).first())
if empty:
    r4 = c.post(f'/api/save_recurring_template/{empty.id}', json={'label': 'smoketest2'})
    assert r4.status_code == 400, f"empty page should 400, got {r4.status_code}"
    print("ok: saving a page with no ad boxes is rejected (400)")

# cleanup
with app.app_context():
    RecurringPageTemplate.query.filter_by(masthead=mh, label='smoketest').delete()
    db.session.commit()
print("\nALL RECURRING-TEMPLATE SMOKE CHECKS PASSED (local template cleaned up)")
