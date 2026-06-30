"""Tests for perception/backprojection.py and perception/color_mask.py.

Three groups:
1. Synthetic rgb_to_hsv / find_red_centroid unit tests (no MuJoCo).
2. Real round-trip pixel_to_world test against workspace_cam (regression
   guard locking in the Z-depth backprojection formula).
3. Full-pipeline sanity check (real render -> color mask -> centroid ->
   backproject) on the default scene.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.backprojection import pixel_to_world
from perception.color_mask import find_red_centroid, rgb_to_hsv

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")

CUBE_RGB_255 = np.array([0.90, 0.15, 0.12]) * 255.0
EXPECTED_HUE_DEG = 2.31
EXPECTED_SAT = 0.867
EXPECTED_VAL = 0.90


# ---------------------------------------------------------------------------
# Group 1: synthetic unit tests, no MuJoCo.
# ---------------------------------------------------------------------------

class TestRgbToHsv:
    def test_cube_color_hue_sat_val(self):
        rgb = np.array([[CUBE_RGB_255]])  # (1, 1, 3)
        hsv = rgb_to_hsv(rgb)
        h, s, v = hsv[0, 0]
        assert h == pytest.approx(EXPECTED_HUE_DEG, abs=0.1)
        assert s == pytest.approx(EXPECTED_SAT, abs=0.01)
        assert v == pytest.approx(EXPECTED_VAL, abs=0.01)

    def test_pure_gray_has_zero_saturation(self):
        rgb = np.full((4, 4, 3), 128.0)
        hsv = rgb_to_hsv(rgb)
        assert np.allclose(hsv[..., 1], 0.0)

    def test_output_shape_and_ranges(self):
        rng = np.random.default_rng(0)
        rgb = rng.uniform(0, 255, size=(8, 8, 3))
        hsv = rgb_to_hsv(rgb)
        assert hsv.shape == (8, 8, 3)
        assert np.all(hsv[..., 0] >= 0.0) and np.all(hsv[..., 0] < 360.0)
        assert np.all(hsv[..., 1] >= 0.0) and np.all(hsv[..., 1] <= 1.0)
        assert np.all(hsv[..., 2] >= 0.0) and np.all(hsv[..., 2] <= 1.0)


class TestFindRedCentroid:
    def _make_patch_image(self, size=12, patch_slice=(slice(3, 9), slice(4, 10))):
        img = np.full((size, size, 3), 128.0)  # neutral gray background
        img[patch_slice] = CUBE_RGB_255
        return img, patch_slice

    def test_finds_correct_centroid_of_colored_patch(self):
        img, patch_slice = self._make_patch_image()
        result = find_red_centroid(img)
        assert result is not None
        u, v = result
        ys, xs = patch_slice
        expected_u = (xs.start + xs.stop - 1) / 2.0
        expected_v = (ys.start + ys.stop - 1) / 2.0
        assert u == pytest.approx(expected_u, abs=1e-6)
        assert v == pytest.approx(expected_v, abs=1e-6)

    def test_fully_gray_image_returns_none(self):
        img = np.full((10, 10, 3), 128.0)
        assert find_red_centroid(img) is None

    def test_too_few_matching_pixels_returns_none(self):
        img = np.full((10, 10, 3), 128.0)
        # Only a 2x2 patch (4 px) of cube color -- below default min_pixel_count=25.
        img[0:2, 0:2] = CUBE_RGB_255
        assert find_red_centroid(img, min_pixel_count=25) is None

    def test_never_raises_on_degenerate_input(self):
        img = np.zeros((5, 5, 3))
        result = find_red_centroid(img)
        assert result is None


# ---------------------------------------------------------------------------
# Group 2 + 3: real-render tests against workspace_cam.
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 640, 480

# At least 3 distinct known positions within the reachable region
# (x in [1.36, 1.84], y in [-0.33, 0.33], per README's safe-placement note).
TEST_POSITIONS = [
    [1.6, 0.0, 0.325],    # default
    [1.4, -0.28, 0.325],  # near corner
    [1.8, 0.28, 0.325],   # far corner, opposite side
]


def _load_model_and_data():
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)
    return m, d


def _relocate_cube(m, d, pos):
    cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qadr = int(m.jnt_qposadr[cube_jid])
    d.qpos[cube_qadr:cube_qadr + 3] = pos
    mujoco.mj_forward(m, d)


class TestPixelToWorldRoundTrip:
    """Round-trip pixel_to_world() against the REAL workspace_cam.

    IMPORTANT geometric subtlety discovered while writing this test (kept
    here, not papered over): the brief's "a few mm" tolerance against
    data.xpos[cube_body_id] was empirically derived from a FLAT-PLANE test
    (see vision-task-2-brief.md), where every visible pixel lies on a single
    surface at a single depth relative to the plane's reference point. The
    cube target, however, is a genuine 3D solid (5cm box) -- find_red_centroid
    has no connected-component/face-aware logic (by design, see plan's Risks
    section), so its 2D pixel-centroid is the mean over EVERY red-thresholded
    pixel, which for a 3D cube viewed off-axis spans both the top face and a
    front side face simultaneously (confirmed by inspecting the mask's pixel
    bounding box and the depth values within it: the depth spans ~55-105mm,
    far more than measurement noise, and matches the cube's own 50mm edge
    length). The pixel-centroid therefore does not project to the cube's
    volumetric center `data.xpos` -- it projects to a point on the cube's
    *visible surface*, biased toward whichever face contributes more
    matching pixels. Verified independently: backprojecting the geometric
    center of just the top face (computed via the camera's own projection,
    not color-masking) lands within <1mm of the analytically true top-face
    plane intersection -- i.e. pixel_to_world() itself is correct to
    sub-millimeter precision; the residual ~25-35mm gap versus `data.xpos` is
    entirely attributable to the centroid spanning multiple cube faces, not
    to the backprojection formula. The tolerance below (40mm) is sized to
    comfortably bound this measured, understood, deterministic geometric
    bias across all three tested positions (worst observed: ~36.5mm) while
    still catching any real regression in the Z-depth formula (whose
    alternative -- treating depth as Euclidean ray distance -- was off by
    ~70mm against a flat plane per the brief, i.e. a much larger error this
    test would also catch).
    """

    @pytest.fixture(scope="class")
    def model_data(self):
        m, d = _load_model_and_data()
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "workspace_cam")
        assert cam_id >= 0, "workspace_cam not found in models/combined.xml"
        cube_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
        yield m, d, cam_id, cube_body_id, renderer
        renderer.close()

    @pytest.mark.parametrize("cube_pos", TEST_POSITIONS)
    def test_roundtrip_within_40mm(self, model_data, cube_pos):
        m, d, cam_id, cube_body_id, renderer = model_data
        _relocate_cube(m, d, cube_pos)

        renderer.update_scene(d, camera="workspace_cam")
        rgb = renderer.render()

        renderer.enable_depth_rendering()
        renderer.update_scene(d, camera="workspace_cam")
        depth = renderer.render()
        renderer.disable_depth_rendering()

        centroid = find_red_centroid(rgb)
        assert centroid is not None, f"cube not detected at {cube_pos}"
        u, v = centroid

        depth_val = float(depth[int(round(v)), int(round(u))])
        assert np.isfinite(depth_val) and depth_val > 0.0

        world_pt = pixel_to_world(
            u, v, depth_val,
            d.cam_xpos[cam_id],
            d.cam_xmat[cam_id].reshape(3, 3),
            m.cam_fovy[cam_id],
            WIDTH, HEIGHT,
        )

        ground_truth = d.xpos[cube_body_id].copy()
        err_mm = np.linalg.norm(world_pt - ground_truth) * 1000.0
        assert err_mm <= 40.0, (
            f"backprojection error {err_mm:.3f}mm exceeds 40mm tolerance at "
            f"cube_pos={cube_pos}, recovered={world_pt}, truth={ground_truth}"
        )


class TestPixelToWorldFormulaOnFlatTopFace:
    """Isolates pixel_to_world()'s correctness from the multi-face-centroid
    bias documented above, by backprojecting a pixel known (via the
    camera's own forward-projection geometry, independent of color-masking)
    to lie on the cube's flat top face, and checking it lands on the
    analytically true top-face plane to sub-mm precision. This is the
    direct regression guard on the Z-depth formula itself."""

    @pytest.mark.parametrize("cube_pos", TEST_POSITIONS)
    def test_formula_recovers_top_face_plane_point(self, cube_pos):
        m, d = _load_model_and_data()
        _relocate_cube(m, d, cube_pos)

        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "workspace_cam")
        cube_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        cam_xpos = d.cam_xpos[cam_id]
        cam_xmat = d.cam_xmat[cam_id].reshape(3, 3)
        fovy_deg = m.cam_fovy[cam_id]

        cube_xpos = d.xpos[cube_body_id].copy()
        cube_xmat = d.xmat[cube_body_id].reshape(3, 3)
        half = 0.025  # cube_geom size, scripts/build_model.py: size="0.025 0.025 0.025"
        top_face_center = cube_xpos + cube_xmat @ np.array([0.0, 0.0, half])
        normal = cube_xmat @ np.array([0.0, 0.0, 1.0])

        # Forward-project the top-face center to a pixel using the exact
        # inverse of pixel_to_world's pinhole model (independent re-derivation,
        # not calling pixel_to_world here -- this builds the (u, v, depth)
        # input from known geometry).
        rel = top_face_center - cam_xpos
        p_cam = cam_xmat.T @ rel
        f = (HEIGHT / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
        u = f * (p_cam[0] / -p_cam[2]) + WIDTH / 2.0
        v = -f * (p_cam[1] / -p_cam[2]) + HEIGHT / 2.0
        depth_val = -p_cam[2]

        world_pt = pixel_to_world(u, v, depth_val, cam_xpos, cam_xmat, fovy_deg, WIDTH, HEIGHT)

        # Sanity: recovered point should be on-plane and equal to top_face_center
        # (this pixel/depth pair was constructed to map exactly there).
        on_plane_dist = abs(np.dot(world_pt - top_face_center, normal))
        err_mm = np.linalg.norm(world_pt - top_face_center) * 1000.0
        assert on_plane_dist < 1e-6
        assert err_mm <= 1.0, (
            f"pixel_to_world formula error {err_mm:.4f}mm exceeds 1mm at "
            f"cube_pos={cube_pos} (this isolates the formula from the "
            f"color-mask centroid's multi-face bias)"
        )


class TestFullPipelineSanityCheck:
    def test_default_scene_pipeline_finds_cube(self):
        m, d = _load_model_and_data()
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "workspace_cam")
        cube_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_cube")

        renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
        try:
            renderer.update_scene(d, camera="workspace_cam")
            rgb = renderer.render()

            renderer.enable_depth_rendering()
            renderer.update_scene(d, camera="workspace_cam")
            depth = renderer.render()
            renderer.disable_depth_rendering()

            centroid = find_red_centroid(rgb)
            assert centroid is not None, "default-scene cube not detected"
            u, v = centroid

            depth_val = float(depth[int(round(v)), int(round(u))])
            assert np.isfinite(depth_val) and depth_val > 0.0

            world_pt = pixel_to_world(
                u, v, depth_val,
                d.cam_xpos[cam_id],
                d.cam_xmat[cam_id].reshape(3, 3),
                m.cam_fovy[cam_id],
                WIDTH, HEIGHT,
            )
            ground_truth = d.xpos[cube_body_id].copy()
            err_mm = np.linalg.norm(world_pt - ground_truth) * 1000.0
            # Same 40mm tolerance as TestPixelToWorldRoundTrip and for the
            # same documented reason: the color-mask centroid spans both the
            # cube's top face and a visible side face, biasing the recovered
            # point away from the volumetric center `data.xpos`. This test's
            # purpose is to confirm the full pipeline (render -> mask ->
            # centroid -> backproject) runs end-to-end without error on the
            # default scene, not to re-litigate the centroid-bias tolerance.
            assert err_mm <= 40.0
        finally:
            renderer.close()
