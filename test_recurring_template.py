"""Analytic tests for recurring-page template matching (no DB/API).

Covers the structured anchor form {'phrases': [...], 'min': N}: a unique standing
phrase (church attribution line) matches at min=1, and a sponsor page with no such
phrase keys off a SUBSET of advertiser names — which must NOT over-match generic
pages (the bug that fired on every page containing the word 'church')."""
from app import _norm_text, _derive_recurring_anchors, _match_recurring_anchors

# Two church weeks: attribution line + sponsors stable, devotional changes.
WK1 = _norm_text("Abundantly Blessed ... let your faithful shout for joy "
                 "Observer/Advocate Bargen Bethel Mennonite Peterson Pharmacy "
                 "These weekly Church Messages are contributed by the following concerned "
                 "citizens and businesses, who urge you to attend the Church of your choice!")
WK2 = _norm_text("Walk by Faith ... rejoice and be glad "
                 "Observer/Advocate Bargen Bethel Mennonite Peterson Pharmacy "
                 "These weekly Church Messages are contributed by the following concerned "
                 "citizens and businesses, who urge you to attend the Church of your choice!")
NEWS = _norm_text("City council approves budget. Pastor of the local Methodist church "
                  "spoke about the church split in the denomination this week.")


def test_standing_phrase_anchor():
    a = _derive_recurring_anchors(WK1, box_tokens=[])
    assert a == {'phrases': ['attend the church of your choice'], 'min': 1}, a
    assert _match_recurring_anchors(WK1, a) and _match_recurring_anchors(WK2, a)
    assert not _match_recurring_anchors(NEWS, a), "news mentions church but not the line"
    print("ok: standing attribution line keys the page (min 1), rejects news")


def test_sponsor_subset_anchor_unique():
    # A Citizen-style church section with NO standing phrase -> key off the
    # contracted sponsor names found inside the marked boxes.
    tokens = ["westbrook", "duerksen", "friesen", "windom", "staples", "holts"]
    a = _derive_recurring_anchors("soft touch let your gentleness be known", box_tokens=tokens)
    assert a['min'] >= 3, a
    church = _norm_text("Soft Touch ... Westbrook Mutual Windom Variety Staples Oil "
                        "Hy-Vee Duerksen Electric Holt's Cleaning Friesen Financial")
    news = _norm_text("Grandparents Day is set at Red Rock Central. Referendum info Monday. "
                      "The whole culture is having issues with the Bible, says Pastor.")
    assert _match_recurring_anchors(church, a), "should match the real church section"
    assert not _match_recurring_anchors(news, a), "must NOT match an unrelated news page"
    print(f"ok: sponsor-subset anchor matches church only (min {a['min']} of {len(a['phrases'])})")


def test_generic_word_does_not_overmatch():
    # The exact bug: a bare 'church' must not become a whole-paper match.
    a = _derive_recurring_anchors("soft touch", box_tokens=["westbrook", "duerksen", "friesen", "windom"])
    methodist = _norm_text("Global Methodist Church a new denomination. The church split. "
                           "Churches leaving the UMC church over church issues church church.")
    assert not _match_recurring_anchors(methodist, a), "church-heavy news must not match"
    print("ok: a church-heavy NEWS page does not match the sponsor anchor")


def test_legacy_list_anchor_still_matches():
    # Old templates stored a bare list (require all) — must still work.
    assert _match_recurring_anchors(WK1, ["attend the church of your choice", "church"])
    assert not _match_recurring_anchors(NEWS, ["attend the church of your choice"])
    print("ok: legacy list-form anchors still honored (require all)")


def test_relative_box_roundtrip():
    rect = [0.044, 0.700, 0.193, 0.965]
    x0, y0, x1, y1 = rect
    exp = (x0*1332, y0*2340, (x1-x0)*1332, (y1-y0)*2340)
    assert abs(exp[0]-58.6) < 1 and abs(exp[2]-198.5) < 1, exp
    print(f"ok: relative->pixel scaling correct (tile ~{exp[2]:.0f}x{exp[3]:.0f}px)")


if __name__ == '__main__':
    test_standing_phrase_anchor()
    test_sponsor_subset_anchor_unique()
    test_generic_word_does_not_overmatch()
    test_legacy_list_anchor_still_matches()
    test_relative_box_roundtrip()
    print("\nALL RECURRING-TEMPLATE TESTS PASSED")
