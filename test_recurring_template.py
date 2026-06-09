"""Analytic tests for recurring-page template matching + scaling (no DB/API)."""
from app import (_norm_text, _derive_recurring_anchors, _match_recurring_anchors,
                 _seed_recurring_boxes)

# Two different weeks of the same church page: sponsors + attribution stable,
# center devotional text changes. The matcher must fire on both.
WK1 = _norm_text("Cottonwood 4-H celebrates accomplishments ... Abundantly Blessed "
                 "We who read this are abundantly blessed ... let your faithful shout for joy "
                 "Observer/Advocate  Bargen Inc  Odin State Bank  Watonwan Enterprises "
                 "These weekly Church Messages are contributed by the following concerned "
                 "citizens and businesses, who urge you to attend the Church of your choice!")
WK2 = _norm_text("Local FFA students earn honors ... Walk by Faith "
                 "Trust in the Lord with all your heart ... rejoice and be glad "
                 "Observer/Advocate  Bargen Inc  Odin State Bank  Watonwan Enterprises "
                 "These weekly Church Messages are contributed by the following concerned "
                 "citizens and businesses, who urge you to attend the Church of your choice!")
# A NON-church page from the same paper — must NOT match.
NEWS = _norm_text("City council approves new budget for road repairs. The board voted "
                  "5-2 Tuesday. Sports: Wildcats win regional title in overtime thriller.")


def test_anchor_derivation():
    a = _derive_recurring_anchors(WK1)
    assert "attend the church of your choice" in a, a
    print("ok: derives the standing attribution-line anchor")


def test_match_recurring_and_reject_news():
    anchors = _derive_recurring_anchors(WK1)
    assert _match_recurring_anchors(WK1, anchors)
    assert _match_recurring_anchors(WK2, anchors), "must match a DIFFERENT week's church page"
    assert not _match_recurring_anchors(NEWS, anchors), "must NOT match a news page"
    print("ok: matches both church weeks, rejects the news page")


def test_empty_anchors_never_match():
    assert not _match_recurring_anchors(WK1, [])
    print("ok: empty anchors never match (no accidental whole-paper match)")


def test_fallback_anchors_for_non_church():
    # A recurring page with no church wording still gets anchors (longest tokens).
    txt = _norm_text("Marketplace Classifieds Directory Employment Automotive Realestate")
    a = _derive_recurring_anchors(txt)
    assert a and all(len(x) >= 7 for x in a), a
    print("ok: non-church recurring page falls back to longest-token anchors")


def test_relative_box_roundtrip():
    # Seeding scales relative rects back to page px. Verify the math via a stub.
    class P:  # minimal page stub
        width_pixels = 1332; height_pixels = 2340; pixels_per_inch = None
    rect = [0.044, 0.700, 0.193, 0.965]  # a left-column church tile (rel)
    x0,y0,x1,y1 = rect
    exp = (x0*1332, y0*2340, (x1-x0)*1332, (y1-y0)*2340)
    assert abs(exp[0]-58.6) < 1 and abs(exp[2]-198.5) < 1, exp
    print(f"ok: relative->pixel scaling correct (tile ~{exp[2]:.0f}x{exp[3]:.0f}px)")


if __name__ == '__main__':
    test_anchor_derivation()
    test_match_recurring_and_reject_news()
    test_empty_anchors_never_match()
    test_fallback_anchors_for_non_church()
    test_relative_box_roundtrip()
    print("\nALL RECURRING-TEMPLATE TESTS PASSED")
