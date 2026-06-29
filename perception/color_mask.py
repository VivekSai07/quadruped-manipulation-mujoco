"""Pure-numpy HSV color thresholding + centroid extraction. No cv2 dependency."""
from __future__ import annotations
import numpy as np


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb: (H, W, 3) uint8 or float in [0,255]. Returns (H, W, 3) float HSV,
    H in [0, 360), S and V in [0, 1]. Vectorized, no cv2."""
    arr = rgb.astype(np.float64) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.where(maxc == 0, 1, maxc), 0.0)

    rc = np.where(delta > 0, ((g - b) / np.where(delta == 0, 1, delta)) % 6, 0.0)
    gc = np.where(delta > 0, ((b - r) / np.where(delta == 0, 1, delta)) + 2.0, 0.0)
    bc = np.where(delta > 0, ((r - g) / np.where(delta == 0, 1, delta)) + 4.0, 0.0)
    h = np.where(delta == 0, 0.0, np.where(maxc == r, rc, np.where(maxc == g, gc, bc)))
    h = (h * 60.0) % 360.0
    return np.stack([h, s, v], axis=-1)


def find_red_mask(
    rgb: np.ndarray,
    hue_center_deg: float = 2.31,
    hue_tolerance_deg: float = 18.0,
    min_saturation: float = 0.5,
    min_value: float = 0.5,
) -> np.ndarray:
    """Return a boolean (H, W) mask of pixels matching the red threshold."""
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_dist = np.minimum(np.abs(h - hue_center_deg), 360.0 - np.abs(h - hue_center_deg))
    return (hue_dist <= hue_tolerance_deg) & (s >= min_saturation) & (v >= min_value)


def find_red_centroid(
    rgb: np.ndarray,
    hue_center_deg: float = 2.31,
    hue_tolerance_deg: float = 18.0,
    min_saturation: float = 0.5,
    min_value: float = 0.5,
    min_pixel_count: int = 25,
) -> tuple[float, float] | None:
    """Return (u, v) pixel centroid of red-thresholded pixels, or None if
    fewer than min_pixel_count pixels match (signal clean failure -- never
    raises, never returns a degenerate/garbage centroid)."""
    mask = find_red_mask(
        rgb,
        hue_center_deg=hue_center_deg,
        hue_tolerance_deg=hue_tolerance_deg,
        min_saturation=min_saturation,
        min_value=min_value,
    )

    ys, xs = np.nonzero(mask)
    if xs.size < min_pixel_count:
        return None
    return float(xs.mean()), float(ys.mean())
