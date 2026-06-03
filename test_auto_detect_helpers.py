"""Pure-function tests for the auto-detection orchestrator helpers.

These don't touch the database or Flask app — they validate the IoU /
containment / dedup math that powers cross-tier deduplication. Run with:
    python test_auto_detect_helpers.py
"""
import sys


def _box(x, y, w, h):
    return {'x': x, 'y': y, 'width': w, 'height': h}


def run():
    # Import after arg-parse so failures are clearer.
    from app import _iou, _containment, _dedupe_against, _demote_containers
    from ad_judge import parse_response, VERDICT_AD, VERDICT_EDITORIAL, VERDICT_FURNITURE
    from pdf_structure_analyzer import PDFStructureAdDetector

    failures = []

    def check(name, cond):
        if cond:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}")
            failures.append(name)

    # --- _iou ---
    a = _box(0, 0, 100, 100)
    b = _box(0, 0, 100, 100)
    check("identical boxes -> IoU == 1.0", abs(_iou(a, b) - 1.0) < 1e-9)

    a = _box(0, 0, 100, 100)
    b = _box(200, 200, 50, 50)
    check("disjoint boxes -> IoU == 0", _iou(a, b) == 0.0)

    a = _box(0, 0, 100, 100)
    b = _box(50, 0, 100, 100)
    # inter = 50*100 = 5000; union = 10000 + 10000 - 5000 = 15000
    check("half overlap -> IoU == 1/3", abs(_iou(a, b) - (5000 / 15000)) < 1e-9)

    a = _box(0, 0, 100, 100)
    b = _box(10, 10, 20, 20)
    # inter = 400; union = 10000 + 400 - 400 = 10000; IoU = 0.04
    check("nested small inside large -> IoU small", abs(_iou(a, b) - 0.04) < 1e-9)

    # --- _containment ---
    check("containment: small inside large -> 1.0",
          abs(_containment(a, b) - 1.0) < 1e-9)

    a = _box(0, 0, 100, 100)
    b = _box(50, 50, 100, 100)
    # inter = 50*50 = 2500; min(area_a, area_b) = 10000; containment = 0.25
    check("containment: quarter overlap -> 0.25",
          abs(_containment(a, b) - 0.25) < 1e-9)

    # --- _dedupe_against ---
    accepted = [_box(0, 0, 100, 100)]

    # candidate overlapping too much -> dropped
    cands = [_box(10, 10, 100, 100)]  # IoU > 0.35
    kept = _dedupe_against(cands, accepted)
    check("dedup drops heavily-overlapping candidate", len(kept) == 0)

    # candidate far away -> kept
    cands = [_box(500, 500, 100, 100)]
    kept = _dedupe_against(cands, accepted)
    check("dedup keeps disjoint candidate", len(kept) == 1)

    # candidate fully inside accepted -> dropped via containment
    cands = [_box(10, 10, 20, 20)]
    kept = _dedupe_against(cands, accepted)
    check("dedup drops fully-contained candidate", len(kept) == 0)

    # zero-area candidate is skipped
    cands = [_box(200, 200, 0, 0), _box(300, 300, 100, 100)]
    kept = _dedupe_against(cands, accepted)
    check("dedup skips zero-area candidate, keeps valid one", len(kept) == 1)

    # --- sort-by-area dedup: outer frame must beat inner panel ---
    # Mimics the Triumph State Bank case: a whole-ad rectangle that contains
    # a smaller LOCATIONS sub-panel. Both are "candidates"; we must keep only
    # the outer one.
    outer = _box(100, 100, 800, 300)   # whole ad
    inner = _box(700, 120, 180, 250)   # LOCATIONS panel, contained in outer
    # If caller passes inner first, sort-by-area-desc in _dedupe_against
    # must still prefer outer.
    kept = _dedupe_against([inner, outer], accepted=[])
    check("dedupe keeps outer frame over contained inner panel (biggest-first)",
          len(kept) == 1 and kept[0] is outer)

    # Reverse order: still biggest-first.
    kept = _dedupe_against([outer, inner], accepted=[])
    check("dedupe: stable regardless of input order",
          len(kept) == 1 and kept[0] is outer)

    # Two disjoint big ads + one inner contained in the first: keep both bigs.
    other_big = _box(1200, 100, 400, 300)
    kept = _dedupe_against([inner, outer, other_big], accepted=[])
    check("dedupe: two disjoint outer ads kept, inner dropped",
          len(kept) == 2 and outer in kept and other_big in kept)

    # Sub-panel rule: a small panel whose center sits inside a much larger ad,
    # but which only partially overlaps (so containment < 0.80), must still be
    # dropped as a decorative sub-region. Models the nested house-ad case.
    parent = _box(100, 100, 800, 1200)   # area 960,000
    # sub-panel is 300x200 = 60,000 (6% of parent). Center lies inside parent
    # but it sticks out 20px above parent's top edge, so containment is < 1.0.
    sub_panel = _box(200, 80, 300, 200)  # top at y=80, parent top at y=100
    kept = _dedupe_against([sub_panel, parent], accepted=[])
    check("dedupe: sub-panel (center inside, area<50%) dropped via center rule",
          len(kept) == 1 and kept[0] is parent)

    # Two side-by-side ads of similar size: neither's center sits inside the
    # other, so both survive even when they touch at the edges.
    left_ad = _box(0, 0, 400, 300)
    right_ad = _box(420, 0, 400, 300)  # 20px gap, neither contains the other
    kept = _dedupe_against([left_ad, right_ad], accepted=[])
    check("dedupe: two sibling ads side-by-side both survive",
          len(kept) == 2)

    # --- _demote_containers: section/page frames drop, distinct ads survive ---
    # A directory frame enclosing 4 small business-card ads must be dropped so
    # the cards survive dedupe instead of collapsing into the frame.
    frame = _box(0, 0, 1000, 800)          # big section border
    card_a = _box(20, 20, 200, 150)
    card_b = _box(240, 20, 200, 150)
    card_c = _box(20, 200, 200, 150)
    card_d = _box(240, 200, 200, 150)
    demoted = _demote_containers([frame, card_a, card_b, card_c, card_d])
    check("demote: frame enclosing 4 cards is dropped", frame not in demoted)
    check("demote: all 4 inner cards survive",
          all(c in demoted for c in (card_a, card_b, card_c, card_d)))

    # A single display ad with one decorative sub-panel keeps BOTH (only 1 child
    # < threshold 3), so dedupe can later prefer the outer frame.
    ad = _box(0, 0, 800, 300)
    sub = _box(600, 40, 150, 200)
    demoted2 = _demote_containers([ad, sub])
    check("demote: ad with single sub-panel is NOT demoted", ad in demoted2 and sub in demoted2)

    # End-to-end: frame + 4 cards through demotion then dedupe -> 4 boxes kept.
    kept_dc = _dedupe_against(_demote_containers([frame, card_a, card_b, card_c, card_d]), accepted=[])
    check("demote+dedupe: directory yields 4 card boxes, not 1 frame", len(kept_dc) == 4)

    # --- Claude response parser ---
    # Well-formed 2-image response.
    txt = (
        "IMAGE 1\nVERDICT: AD\nREASON: Display ad with phone and URL.\n"
        "\nIMAGE 2\nVERDICT: EDITORIAL\nREASON: News story with byline."
    )
    pairs = parse_response(txt, expected_count=2)
    check("parse_response: verdicts",
          pairs[0][0] == VERDICT_AD and pairs[1][0] == VERDICT_EDITORIAL)
    check("parse_response: reasons non-empty",
          bool(pairs[0][1]) and bool(pairs[1][1]))

    # Model responded with only 1 stanza but we expected 2 -> pad with EDITORIAL.
    txt_short = "IMAGE 1\nVERDICT: AD\nREASON: Looks like an ad."
    pairs = parse_response(txt_short, expected_count=2)
    check("parse_response: missing stanza defaults to EDITORIAL",
          pairs[0][0] == VERDICT_AD and pairs[1][0] == VERDICT_EDITORIAL)

    # Garbage response -> all defaulted.
    pairs = parse_response("I cannot see anything in these images.", expected_count=3)
    check("parse_response: unparseable -> all EDITORIAL",
          all(p[0] == VERDICT_EDITORIAL for p in pairs))

    # Verdict case insensitivity + FURNITURE.
    txt_f = "IMAGE 1\nVERDICT: furniture\nREASON: Masthead."
    pairs = parse_response(txt_f, expected_count=1)
    check("parse_response: case-insensitive FURNITURE",
          pairs[0][0] == VERDICT_FURNITURE)

    # --- Content-cluster structural helper ---
    # Two text blocks 10pt apart plus an image 15pt away => one cluster.
    structure = {
        'text_blocks': [
            {'bounds': [100.0, 100.0, 200.0, 150.0], 'text_content': "SALE"},
            {'bounds': [100.0, 160.0, 200.0, 200.0], 'text_content': "Call 555-0100"},
        ],
        'images': [
            {'bounds': [215.0, 100.0, 300.0, 200.0]},
        ],
        'drawings': [],
        'page_rect': None,
    }
    clusters = PDFStructureAdDetector._find_cluster_candidates(structure)
    check("cluster detector: finds 1 cluster from 3 nearby elements",
          len(clusters) == 1)
    if clusters:
        c = clusters[0]
        check("cluster bounds encompass all elements",
              c['x'] <= 100 and c['y'] <= 100 and
              c['x'] + c['width'] >= 300 and c['y'] + c['height'] >= 200)

    # Far-apart elements => no cluster.
    structure2 = {
        'text_blocks': [
            {'bounds': [100.0, 100.0, 150.0, 120.0], 'text_content': "A"},
            {'bounds': [500.0, 500.0, 550.0, 520.0], 'text_content': "B"},
        ],
        'images': [],
        'drawings': [],
        'page_rect': None,
    }
    clusters2 = PDFStructureAdDetector._find_cluster_candidates(structure2)
    check("cluster detector: no cluster when elements are too far apart",
          len(clusters2) == 0)

    if failures:
        print(f"\n{len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("\nAll helper tests passed.")


if __name__ == '__main__':
    run()
