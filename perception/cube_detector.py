"""CubeDetector: render -> color-threshold -> depth-filter to isolate the
cube's top face -> backproject the filtered centroid to world XYZ.

Why the depth filter exists (see .superpowers/sdd/vision-task-2-report.md and
vision-task-3-brief.md for the full diagnostic story): at workspace_cam's
oblique, elevated, downward-looking angle, the camera sees BOTH the cube's
flat top face AND a visible vertical front side face. Both are red and both
get thresholded into the same color mask, but they sit at different depths
(measured: the matched mask's depth values span ~50mm -- almost exactly the
cube's own edge length, not noise). Naively averaging every matched pixel's
(u, v) into one centroid -- as Phase 2's find_red_centroid does, by design --
produces a point that does not correspond to any real point on the cube; it
lands somewhere between the top face's true position and the lower, farther
side face. Phase 2 measured this bias at ~30-36mm.

The fix is geometric, not statistical: the top face is the highest point on
the cube and the camera looks down at it almost perpendicularly, so top-face
pixels are the CLOSEST (smallest Z-depth) among all red-thresholded pixels.
The receding front face is farther away (larger Z-depth). So: find the
minimum depth among all matched pixels, keep only the pixels within a
tolerance of that minimum (isolating the top face's own pixel cluster), and
compute the centroid + a robust (median) representative depth from that
filtered subset only, before backprojecting. This is informed directly by
Phase 2's own diagnostic measurements, not speculative tuning.

Tuning note on `depth_cluster_tolerance`: the brief's starting guess of 0.010
(1cm) assumed the top face's own pixels all sit at nearly the same depth. At
workspace_cam's oblique angle this is NOT quite true -- empirical measurement
(see vision-task-3-report.md) found the TOP FACE ALONE spans ~40mm of depth
across its own four corners (the near corner of the top face is ~40mm closer
to the camera than its far corner, simply from perspective across the table),
while the side face's depth range starts only slightly beyond that, with a
real but modest gap (~5-9mm) separating the two clusters. A 10mm tolerance
therefore only captures a sliver of the top face nearest the camera, which
biases the centroid toward that one corner -- worse than capturing the whole
face. Widening the tolerance to ~0.05 (50mm) reliably captures the full top
face cluster up to (but not past) the real depth gap before the side face
begins, which empirically minimizes the error. Measured plateau: accuracy
stops improving past ~45-50mm and holds flat (~17-20mm error) out to 80mm+,
confirming the value is sitting inside a genuine cluster-separation plateau
rather than on a knife's edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .color_mask import find_red_mask
from .backprojection import pixel_to_world


@dataclass
class CubeDetectorConfig:
    camera_name: str = "workspace_cam"
    width: int = 640
    height: int = 480
    hue_center_deg: float = 2.31
    hue_tolerance_deg: float = 18.0
    min_saturation: float = 0.5
    min_value: float = 0.5
    min_pixel_count: int = 25
    depth_cluster_tolerance: float = 0.050

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CubeDetectorConfig":
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid})


class CubeDetector:
    """Detects a red cube's world-frame position via a fixed MJCF camera.

    Pipeline: render RGB -> color-threshold to a red mask -> render depth ->
    depth-filter the mask down to its nearest-to-camera cluster (the cube's
    top face, see module docstring) -> centroid + median depth of that
    filtered subset -> backproject via pixel_to_world().
    """

    def __init__(self, model: mujoco.MjModel, config: "CubeDetectorConfig | None" = None) -> None:
        self.model = model
        self.cfg = config or CubeDetectorConfig()
        self._cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.cfg.camera_name)
        if self._cam_id < 0:
            raise ValueError(
                f"Camera {self.cfg.camera_name!r} not found in model -- "
                "rebuild it with `python scripts/build_model.py`."
            )
        self._renderer = mujoco.Renderer(model, height=self.cfg.height, width=self.cfg.width)

    def detect(self, data: mujoco.MjData) -> "np.ndarray | None":
        """Detect the cube and return its world-frame (3,) position, or None
        on clean detection failure (too few matched pixels). Never raises."""
        cfg = self.cfg

        self._renderer.update_scene(data, camera=cfg.camera_name)
        rgb = self._renderer.render()

        mask = find_red_mask(
            rgb,
            hue_center_deg=cfg.hue_center_deg,
            hue_tolerance_deg=cfg.hue_tolerance_deg,
            min_saturation=cfg.min_saturation,
            min_value=cfg.min_value,
        )
        if int(mask.sum()) < cfg.min_pixel_count:
            return None

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=cfg.camera_name)
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()

        masked_depths = depth[mask]
        min_depth = float(masked_depths.min())

        # Isolate the cube's top face: keep only pixels within
        # depth_cluster_tolerance of the minimum (nearest-to-camera) depth.
        # At this camera's oblique angle the top face's own pixels span tens
        # of mm of depth (perspective across the table), so the tolerance is
        # wide enough to capture the whole top-face cluster up to the real
        # depth gap that separates it from the farther, receding side face
        # -- see the module docstring's tuning note for the measurements
        # behind this value.
        top_face_mask = mask & (depth <= min_depth + cfg.depth_cluster_tolerance)

        if not top_face_mask.any():
            # Defensive fallback -- shouldn't normally happen since
            # top_face_mask always includes at least the single
            # minimum-depth pixel, but never crash on this.
            top_face_mask = mask

        ys, xs = np.nonzero(top_face_mask)
        u, v = float(xs.mean()), float(ys.mean())
        median_depth = float(np.median(depth[top_face_mask]))

        cam_xpos = data.cam_xpos[self._cam_id]
        cam_xmat = data.cam_xmat[self._cam_id].reshape(3, 3)
        fovy_deg = self.model.cam_fovy[self._cam_id]

        world_pt = pixel_to_world(
            u, v, median_depth,
            cam_xpos, cam_xmat, fovy_deg,
            cfg.width, cfg.height,
        )
        return world_pt

    def close(self) -> None:
        self._renderer.close()
