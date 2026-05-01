"""Unit tests for citylens_core.stages._polygon_smoothing."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from citylens_core.stages._polygon_smoothing import (
    estimate_pixel_tolerance_in_world_units,
    simplify_polygon_coords,
)


def _square(size: float = 10.0, origin=(0.0, 0.0)) -> list[list[float]]:
    x0, y0 = origin
    return [
        [x0, y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0, y0 + size],
        [x0, y0],
    ]


def _is_closed(ring: list[list[float]]) -> bool:
    return len(ring) >= 2 and ring[0][0] == ring[-1][0] and ring[0][1] == ring[-1][1]


def _sawtooth_top_square(
    size: float = 10.0, teeth: int = 30, amp: float = 0.4
) -> list[list[float]]:
    pts: list[list[float]] = []
    pts.append([0.0, 0.0])
    pts.append([size, 0.0])
    pts.append([size, size])
    for i in range(teeth, 0, -1):
        x = size * (i / (teeth + 1))
        y = size + (amp if i % 2 == 0 else -amp)
        pts.append([x, y])
    pts.append([0.0, size])
    pts.append([0.0, 0.0])
    return pts


def test_identity_clean_square_unchanged():
    sq = _square(10.0)
    out = simplify_polygon_coords([sq], tolerance=0.5)

    assert len(out) == 1
    assert len(out[0]) == 5
    assert _is_closed(out[0])
    assert Polygon(out[0]).area == pytest.approx(100.0)


def test_sawtooth_collapses_to_clean_rectangle():
    """At tolerance well above the saw-tooth amplitude the polygon should
    collapse meaningfully. With preserve_topology=True shapely is
    conservative about how aggressive it gets, so we assert "vertex
    count dropped substantially" rather than a specific final count."""
    sawtooth = _sawtooth_top_square(size=10.0, teeth=30, amp=0.4)
    before = len(sawtooth)
    assert before > 30

    out = simplify_polygon_coords([sawtooth], tolerance=2.0)

    assert len(out) == 1
    ring = out[0]
    assert _is_closed(ring)
    # Tolerance 2.0 vs amp 0.4 — most teeth should disappear.
    assert len(ring) < before / 2, f"expected major reduction, got {before} -> {len(ring)}"
    assert Polygon(ring).area == pytest.approx(100.0, abs=10.0)


def test_polygon_with_hole_preserved():
    outer = _square(100.0)
    hole = _square(30.0, origin=(35.0, 35.0))
    out = simplify_polygon_coords([outer, hole], tolerance=0.5)

    assert len(out) == 2
    assert _is_closed(out[0])
    assert _is_closed(out[1])
    poly = Polygon(out[0], [out[1]])
    assert poly.is_valid
    assert poly.area == pytest.approx(10000.0 - 900.0, abs=20.0)


def test_hole_with_preserve_topology_stays_valid():
    """preserve_topology=True intentionally keeps holes — the alternative
    risks producing invalid geometry. We assert the result stays valid
    rather than expecting the hole to disappear."""
    outer = _square(100.0)
    tiny_hole = _square(1.0, origin=(50.0, 50.0))
    out = simplify_polygon_coords([outer, tiny_hole], tolerance=5.0)

    assert len(out) >= 1
    for ring in out:
        assert _is_closed(ring)
    poly = Polygon(out[0], out[1:] if len(out) > 1 else [])
    assert poly.is_valid
    # Outer dominates the area regardless of whether the hole survived.
    assert poly.area > 9900.0


def test_degenerate_three_point_ring_returned_unchanged():
    bad = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    out = simplify_polygon_coords([bad], tolerance=0.5)
    assert out == [bad]


def test_empty_rings_list_returns_empty():
    assert simplify_polygon_coords([], tolerance=0.5) == []


def test_zero_tolerance_is_noop():
    sq = _square(10.0)
    out = simplify_polygon_coords([sq], tolerance=0.0)
    assert len(out) == 1
    assert len(out[0]) == 5
    assert _is_closed(out[0])
    for a, b in zip(out[0], sq):
        assert a[0] == pytest.approx(b[0])
        assert a[1] == pytest.approx(b[1])


def test_estimate_pixel_tolerance_with_metric_transform():
    from affine import Affine

    tr = Affine(0.5, 0.0, 100.0, 0.0, -0.5, 200.0)
    tol = estimate_pixel_tolerance_in_world_units(tr, pixel_tolerance=0.5)
    assert tol == pytest.approx(0.25)


def test_estimate_pixel_tolerance_identity_returns_input():
    from affine import Affine

    tr = Affine.identity()
    tol = estimate_pixel_tolerance_in_world_units(tr, pixel_tolerance=0.5)
    assert tol == pytest.approx(0.5)


def test_estimate_pixel_tolerance_none_transform():
    tol = estimate_pixel_tolerance_in_world_units(None, pixel_tolerance=0.7)
    assert tol == pytest.approx(0.7)


def test_output_rings_always_closed():
    outer = _square(50.0)
    hole = _square(10.0, origin=(20.0, 20.0))
    out = simplify_polygon_coords([outer, hole], tolerance=0.3)
    for ring in out:
        assert _is_closed(ring), f"ring not closed: {ring[:2]} ... {ring[-2:]}"


def test_self_intersecting_input_does_not_crash():
    bowtie = [
        [0.0, 0.0],
        [10.0, 10.0],
        [10.0, 0.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ]
    out = simplify_polygon_coords([bowtie], tolerance=0.5)
    assert isinstance(out, list)
    assert len(out) >= 1
    assert _is_closed(out[0])


def test_malformed_hole_returns_input_unchanged():
    outer = _square(100.0)
    bad_hole = [[10.0, 10.0], [20.0, 10.0]]
    rings = [outer, bad_hole]
    out = simplify_polygon_coords(rings, tolerance=0.5)
    assert out == rings


def test_simplification_actually_reduces_vertex_count():
    sawtooth = _sawtooth_top_square(size=20.0, teeth=40, amp=0.3)
    before = len(sawtooth)
    out = simplify_polygon_coords([sawtooth], tolerance=0.5)
    after = len(out[0])
    assert after < before / 3, f"expected major reduction, got {before} -> {after}"
