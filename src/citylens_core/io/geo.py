from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry

__all__ = [
    "binary_close",
    "binary_dilate",
    "binary_erode",
    "binary_open",
    "binary_mask_stats",
    "clean_binary_mask",
    "geojson_crs_hint",
    "load_geojson_geometries",
    "load_geojson_mask",
    "mask_f1",
    "mask_iou",
    "remove_small_components",
]


def _as_bool_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask).astype(bool)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    return arr


def binary_mask_stats(mask: Any) -> dict[str, int | float]:
    """Return compact coverage/component QA for a binary mask.

    Component areas come from pixel-space polygons emitted by rasterio, so
    this avoids allocating a full integer label grid merely for diagnostics.
    """
    arr = _as_bool_mask(mask)
    pixel_count = int(arr.sum())
    component_count = 0
    largest_component_pixels = 0
    if pixel_count:
        for geom, value in shapes(
            arr.astype(np.uint8),
            mask=arr,
            transform=Affine.identity(),
        ):
            if int(value) != 1:
                continue
            component_count += 1
            try:
                area = int(round(float(shapely_shape(geom).area)))
            except Exception:
                area = 0
            largest_component_pixels = max(largest_component_pixels, area)

    total_pixels = int(arr.size)
    return {
        "pixels": pixel_count,
        "coverage_fraction": (
            float(pixel_count) / float(total_pixels) if total_pixels else 0.0
        ),
        "component_count": component_count,
        "largest_component_pixels": largest_component_pixels,
        "largest_component_fraction": (
            float(largest_component_pixels) / float(total_pixels)
            if total_pixels
            else 0.0
        ),
    }


def load_geojson_geometries(path: Path) -> list[BaseGeometry]:
    data = json.loads(Path(path).read_text())
    geometries: list[BaseGeometry] = []

    def _append_geometry(geometry: Any) -> None:
        if not isinstance(geometry, dict):
            return
        try:
            geom = shapely_shape(geometry)
        except Exception:
            return
        if geom is not None and not geom.is_empty:
            geometries.append(geom)

    if isinstance(data, dict):
        if data.get("type") == "FeatureCollection":
            for feature in data.get("features") or []:
                if isinstance(feature, dict):
                    _append_geometry(feature.get("geometry"))
        elif data.get("type") == "Feature":
            _append_geometry(data.get("geometry"))
        elif "type" in data and "coordinates" in data:
            _append_geometry(data)

    return geometries


def geojson_crs_hint(path: Path) -> str | None:
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    direct = data.get("crs")
    if isinstance(direct, str) and direct.strip():
        return direct.strip().lower()
    if isinstance(direct, dict):
        name = direct.get("properties", {}).get("name")
        if isinstance(name, str) and name.strip():
            return name.strip().lower()

    features = data.get("features") or []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        crs = props.get("crs")
        if isinstance(crs, str) and crs.strip():
            return crs.strip().lower()
    return None


def load_geojson_mask(
    path: Path,
    *,
    out_shape: tuple[int, int],
    transform: Affine | None,
    pixel_space: bool = False,
) -> np.ndarray:
    geometries = load_geojson_geometries(path)
    if not geometries:
        return np.zeros(out_shape, dtype=bool)

    if transform is None:
        if not pixel_space:
            return np.zeros(out_shape, dtype=bool)
        transform = Affine.identity()

    mask_u8 = rasterize(
        shapes=((geom, 1) for geom in geometries),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=False,
        dtype="uint8",
    )
    return mask_u8 > 0


def binary_dilate(mask: Any, radius: int = 1) -> np.ndarray:
    arr = _as_bool_mask(mask)
    radius = int(radius)
    if radius <= 0 or arr.size == 0:
        return arr.copy()

    kernel = (2 * radius) + 1
    padded = np.pad(arr, radius, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    return np.any(windows, axis=(-1, -2))


def binary_erode(mask: Any, radius: int = 1) -> np.ndarray:
    arr = _as_bool_mask(mask)
    radius = int(radius)
    if radius <= 0 or arr.size == 0:
        return arr.copy()

    kernel = (2 * radius) + 1
    padded = np.pad(arr, radius, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    return np.all(windows, axis=(-1, -2))


def binary_open(mask: Any, radius: int = 1) -> np.ndarray:
    return binary_dilate(binary_erode(mask, radius=radius), radius=radius)


def binary_close(mask: Any, radius: int = 1) -> np.ndarray:
    return binary_erode(binary_dilate(mask, radius=radius), radius=radius)


def remove_small_components(mask: Any, min_pixels: int = 1) -> np.ndarray:
    arr = _as_bool_mask(mask)
    min_pixels = int(min_pixels)
    if min_pixels <= 1 or not arr.any():
        return arr.copy()

    selected: list[tuple[Any, int]] = []
    for geom, value in shapes(arr.astype(np.uint8), mask=arr):
        if int(value) != 1:
            continue
        try:
            shapely_geom = shapely_shape(geom)
        except Exception:
            continue
        if shapely_geom.area >= float(min_pixels):
            selected.append((geom, 1))

    if not selected:
        return np.zeros_like(arr, dtype=bool)

    cleaned = rasterize(
        shapes=selected,
        out_shape=arr.shape,
        transform=Affine.identity(),
        fill=0,
        default_value=1,
        all_touched=False,
        dtype="uint8",
    )
    return cleaned > 0


def clean_binary_mask(
    mask: Any,
    *,
    open_radius: int = 1,
    close_radius: int = 1,
    min_component_px: int = 1,
) -> np.ndarray:
    arr = _as_bool_mask(mask)
    cleaned = binary_open(arr, radius=open_radius) if open_radius > 0 else arr.copy()
    cleaned = binary_close(cleaned, radius=close_radius) if close_radius > 0 else cleaned
    return remove_small_components(cleaned, min_pixels=min_component_px)


def mask_iou(a: Any, b: Any) -> float | None:
    lhs = _as_bool_mask(a)
    rhs = _as_bool_mask(b)
    if lhs.shape != rhs.shape:
        raise ValueError("mask shapes must match")
    union = np.logical_or(lhs, rhs)
    union_count = int(union.sum())
    if union_count == 0:
        return 1.0
    inter_count = int(np.logical_and(lhs, rhs).sum())
    return float(inter_count) / float(union_count)


def mask_f1(predicted: Any, reference: Any) -> float | None:
    pred = _as_bool_mask(predicted)
    ref = _as_bool_mask(reference)
    if pred.shape != ref.shape:
        raise ValueError("mask shapes must match")
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, np.logical_not(ref)).sum())
    fn = int(np.logical_and(np.logical_not(pred), ref).sum())
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 1.0
    return float((2 * tp) / denom)
