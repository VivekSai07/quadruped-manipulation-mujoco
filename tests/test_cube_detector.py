"""Tests for perception/cube_detector.py: CubeDetector + CubeDetectorConfig.

Covers:
1. Detection accuracy at >=4 distinct reachable-region positions (target:
   <=2cm vs data.xpos[cube_body_id], achieved via the depth-cluster filter
   that isolates the cube's top face -- see cube_detector.py's module
   docstring for why this is necessary).
2. Graceful failure on two distinct failure modes: cube relocated out of the
   camera's view frustum, and color thresholds that can never match.
3. Renderer cleanup via detector.close() in a fixture teardown.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception import CubeDetector, CubeDetectorConfig

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")

# Reachable region per README's safe-placement note: x in [1.36, 1.84],
# y in [-0.33, 0.33]. Five distinct positions, including off-default ones,
# spanning corners and an off-center point.
TEST_POSITIONS = [
    [1.6, 0.0, 0.325],     # default
    [1.4, -0.28, 0.325],   # near corner
    [1.8, 0.28, 0.325],    # far corner, opposite side
    [1.4, 0.28, 0.325],    # near corner, opposite y
    [1.75, -0.15, 0.325],  # off-center, near-far/right-ish
]

# Far off the table, outside workspace_cam's view frustum.
OUT_OF_FRUSTUM_POS = [5.0, 5.0, 0.325]

ACCURACY_TOLERANCE_MM = 20.0


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


@pytest.fixture(scope="module")
def model_and_cube_id():
    m, d = _load_model_and_data()
    cube_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    return m, d, cube_body_id


@pytest.fixture
def detector(model_and_cube_id):
    m, _, _ = model_and_cube_id
    det = CubeDetector(m)
    try:
        yield det
    finally:
        det.close()


class TestCubeDetectorAccuracy:
    """Accuracy at >=4 distinct reachable-region positions. The depth-cluster
    filter (CubeDetectorConfig.depth_cluster_tolerance) isolates the cube's
    top face from the side face that also appears in the red mask -- see
    cube_detector.py's module docstring. Target: <=2cm, genuinely achieved."""

    @pytest.mark.parametrize("cube_pos", TEST_POSITIONS)
    def test_detection_within_2cm(self, model_and_cube_id, detector, cube_pos):
        m, d, cube_body_id = model_and_cube_id
        _relocate_cube(m, d, cube_pos)

        result = detector.detect(d)
        assert result is not None, f"cube not detected at {cube_pos}"

        ground_truth = d.xpos[cube_body_id].copy()
        err_mm = float(np.linalg.norm(result - ground_truth) * 1000.0)
        assert err_mm <= ACCURACY_TOLERANCE_MM, (
            f"detection error {err_mm:.2f}mm exceeds {ACCURACY_TOLERANCE_MM}mm "
            f"tolerance at cube_pos={cube_pos}, detected={result}, "
            f"truth={ground_truth}"
        )


class TestCubeDetectorGracefulFailure:
    """Two distinct forced-failure modes; neither should raise."""

    def test_returns_none_when_cube_outside_view_frustum(self, model_and_cube_id, detector):
        m, d, _ = model_and_cube_id
        _relocate_cube(m, d, OUT_OF_FRUSTUM_POS)
        result = detector.detect(d)
        assert result is None

    def test_returns_none_when_color_threshold_never_matches(self, model_and_cube_id):
        m, d, _ = model_and_cube_id
        _relocate_cube(m, d, TEST_POSITIONS[0])  # cube visible, normal scene

        # Blue/cyan hue center -- nothing in the scene is this color.
        never_match_cfg = CubeDetectorConfig(hue_center_deg=180.0, hue_tolerance_deg=5.0)
        det = CubeDetector(m, never_match_cfg)
        try:
            result = det.detect(d)
            assert result is None
        finally:
            det.close()


class TestCubeDetectorConfig:
    def test_from_dict_filters_unknown_keys(self):
        cfg = CubeDetectorConfig.from_dict({
            "camera_name": "workspace_cam",
            "depth_cluster_tolerance": 0.02,
            "not_a_real_field": 123,
        })
        assert cfg.camera_name == "workspace_cam"
        assert cfg.depth_cluster_tolerance == 0.02

    def test_defaults(self):
        cfg = CubeDetectorConfig()
        assert cfg.camera_name == "workspace_cam"
        assert cfg.depth_cluster_tolerance == 0.050
        assert cfg.min_pixel_count == 25
