"""Analytic tests for the nested-judge architectural change (no API, no DB).

Exercises the two new helpers that replace the demote->swallow chain:
  _dedupe_keep_nested  -- keeps container AND children for judging
  _reconcile_nested_ads -- post-judge, drops a frame that holds >=2 AD children

Run: python test_nested_reconcile.py
"""
from app import _dedupe_keep_nested, _reconcile_nested_ads, _iou, _containment


def box(x, y, w, h, source='structure_bordered'):
    return {'x': float(x), 'y': float(y), 'width': float(w), 'height': float(h),
            'source': source, 'text_preview': ''}


def _has(boxes, b):
    return any(abs(o['x'] - b['x']) < 1 and abs(o['y'] - b['y']) < 1
               and abs(o['width'] - b['width']) < 1 and abs(o['height'] - b['height']) < 1
               for o in boxes)


# A directory: one big frame + a 2x2 grid of tiles inside it.
FRAME = box(0, 0, 400, 400)
TILES = [box(10, 10, 180, 180), box(210, 10, 180, 180),
         box(10, 210, 180, 180), box(210, 210, 180, 180)]


def test_dedupe_keeps_nested():
    # Old chain dropped the 4 tiles (contained in FRAME). New: all 5 survive to
    # judging -- the tiles are not near-duplicates of the frame.
    kept = _dedupe_keep_nested([FRAME] + TILES, accepted=[])
    assert len(kept) == 5, f"expected frame+4 tiles kept, got {len(kept)}"
    assert _has(kept, FRAME) and all(_has(kept, t) for t in TILES)
    # sanity: a tile is genuinely NOT a near-dup of the frame
    assert _iou(TILES[0], FRAME) < 0.85
    print("ok: _dedupe_keep_nested keeps container + children (5)")


def test_dedupe_collapses_near_duplicates():
    # Same ad boxed by two generators (IoU > 0.85) collapses to the larger one.
    a = box(0, 0, 100, 100, 'structure_bordered')
    b = box(2, 2, 100, 100, 'structure_cluster')
    assert _iou(a, b) > 0.85
    kept = _dedupe_keep_nested([a, b], accepted=[])
    assert len(kept) == 1, f"expected near-dups collapsed, got {len(kept)}"
    print("ok: _dedupe_keep_nested collapses near-duplicate siblings (1)")


def test_dedupe_respects_existing():
    # A candidate duplicating an existing (manual / prior) box is dropped.
    existing = [box(0, 0, 100, 100)]
    cand = box(1, 1, 100, 100)
    kept = _dedupe_keep_nested([cand], accepted=existing)
    assert kept == [], f"expected drop-vs-existing, got {kept}"
    print("ok: _dedupe_keep_nested still drops duplicates of existing boxes (0)")


def test_reconcile_directory_drops_frame():
    # Judge called the frame AND all 4 tiles AD -> frame holds >=2 children ->
    # drop the frame, keep the 4 granular ads (Trevor's rule).
    out = _reconcile_nested_ads([FRAME] + TILES)
    assert len(out) == 4, f"expected 4 inner ads, got {len(out)}"
    assert not _has(out, FRAME), "frame should be dropped"
    assert all(_has(out, t) for t in TILES)
    print("ok: _reconcile_nested_ads drops a multi-ad frame, keeps inner ads (4)")


def test_reconcile_display_ad_kept():
    # A display ad (frame) with a single inner photo, both AD -> 0-1 children ->
    # keep both (merge collapses them downstream; reconcile must not drop them).
    frame = box(0, 0, 300, 300)
    photo = box(20, 20, 120, 120)
    out = _reconcile_nested_ads([frame, photo])
    assert len(out) == 2, f"expected display ad kept whole, got {len(out)}"
    print("ok: _reconcile_nested_ads keeps a 1-child display ad intact (2)")


def test_reconcile_banner_kept_whole():
    # The regression case: a wide banner with 2+ small AD sub-elements that fill
    # only ~5% of it. Coverage guard must KEEP the banner (dropping it would lose
    # the uncovered ~95% of its area, as happened on ccc p8-11 at ~30% coverage).
    banner = box(0, 0, 1000, 400)              # 400,000 px
    logo = box(20, 20, 100, 100)               # inside, small
    caption = box(800, 250, 100, 100)          # inside, small  -> ~5% combined
    out = _reconcile_nested_ads([banner, logo, caption])
    assert _has(out, banner), "banner must be kept whole (sparse inner ads)"
    print("ok: _reconcile_nested_ads keeps a banner whole when inner ads are sparse")


def test_reconcile_directory_threshold_boundary():
    # Tiles filling >=60% of the frame -> still a directory -> drop frame.
    out = _reconcile_nested_ads([FRAME] + TILES)  # 4*180^2 / 400^2 = 81% covered
    assert not _has(out, FRAME) and len(out) == 4
    print("ok: _reconcile_nested_ads still drops a well-filled directory frame (4)")


def test_reconcile_recall_win():
    # The headline case: the frame was judged NON-AD, so it never reaches
    # reconcile. Only the children are passed in -> all survive (no swallow).
    out = _reconcile_nested_ads(TILES)
    assert len(out) == 4, f"expected all children to survive, got {len(out)}"
    print("ok: _reconcile_nested_ads -- rejected frame can't take children down (4)")


def test_reconcile_standalone_unchanged():
    ads = [box(0, 0, 100, 100), box(500, 0, 100, 100), box(0, 500, 100, 100)]
    out = _reconcile_nested_ads(ads)
    assert len(out) == 3
    print("ok: _reconcile_nested_ads leaves non-nested ads alone (3)")


if __name__ == '__main__':
    test_dedupe_keeps_nested()
    test_dedupe_collapses_near_duplicates()
    test_dedupe_respects_existing()
    test_reconcile_directory_drops_frame()
    test_reconcile_display_ad_kept()
    test_reconcile_banner_kept_whole()
    test_reconcile_directory_threshold_boundary()
    test_reconcile_recall_win()
    test_reconcile_standalone_unchanged()
    print("\nALL NESTED-RECONCILE TESTS PASSED")
