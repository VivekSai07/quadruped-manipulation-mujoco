"""Pixel + depth -> 3D world point, pinhole camera model.

MuJoCo camera convention: camera looks down its own local -Z axis; +X is
right, +Y is up in the camera's local frame (OpenGL-style, NOT OpenCV's
+Z-forward/+Y-down convention -- do not mix the two).

Depth semantics (empirically confirmed against mujoco==3.9.0, see
.superpowers/sdd/vision-task-2-brief.md for the verification methodology):
Renderer.render() after enable_depth_rendering() returns Z-DEPTH --
perpendicular distance from the camera's image plane to the point along the
camera's viewing axis -- NOT Euclidean ray distance from the camera to the
point. These are identical for the center pixel but diverge for off-axis
pixels (exactly the case for a detected cube that isn't dead-center in
frame), so getting this right matters.
"""
from __future__ import annotations
import numpy as np


def pixel_to_world(
    u: float, v: float, depth: float,
    cam_xpos: np.ndarray,      # (3,) world position of camera, e.g. data.cam_xpos[cam_id]
    cam_xmat: np.ndarray,      # (3,3) world rotation of camera, e.g. data.cam_xmat[cam_id].reshape(3,3)
    fovy_deg: float,           # model.cam_fovy[cam_id]
    width: int,
    height: int,
) -> np.ndarray:
    """Backproject one pixel + metric Z-depth to a world-frame 3D point."""
    fovy = np.deg2rad(fovy_deg)
    f = (height / 2.0) / np.tan(fovy / 2.0)   # focal length in pixels (vertical)
    cx, cy = width / 2.0, height / 2.0

    x_cam = (u - cx) / f
    y_cam = -(v - cy) / f
    z_cam = -1.0
    ray_cam = np.array([x_cam, y_cam, z_cam])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    # depth is the Z-depth (perpendicular distance), so scale the unit ray
    # so its z-component magnitude equals depth:
    scale = depth / (-ray_cam[2])
    point_cam = ray_cam * scale

    point_world = cam_xpos + cam_xmat @ point_cam
    return point_world
