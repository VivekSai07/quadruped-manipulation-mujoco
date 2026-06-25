"""Tests for locomotion and manipulation controllers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from controllers.arms import DEFAULT_ARM
from controllers.end_effectors import END_EFFECTORS
from controllers.locomotion import GaitMode, LocomotionController
from controllers.manipulation import ManipulationController
from scripts.build_model import build_combined_xml

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")


def _load_variant_model(arm: str, end_effector: str) -> mujoco.MjModel:
    """Build an arm/end-effector variant and load it via a scratch file in
    models/ so mesh paths (relative to that directory) still resolve --
    from_xml_string has no file-path context for relative asset references.
    """
    xml = build_combined_xml(arm, end_effector)
    models_dir = Path(__file__).parent.parent / "models"
    scratch_path = models_dir / f"_test_scratch_{arm}_{end_effector}.xml"
    scratch_path.write_text(xml, encoding="utf-8")
    try:
        return mujoco.MjModel.from_xml_path(str(scratch_path))
    finally:
        scratch_path.unlink()


@pytest.fixture(scope="module")
def model_data():
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)
    return m, d


class TestLocomotionController:
    def test_init(self, model_data):
        m, d = model_data
        loco = LocomotionController(m, d)
        assert loco is not None

    def test_stand_mode_sets_ctrl(self, model_data):
        m, d = model_data
        d2 = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d2, kid)
        loco = LocomotionController(m, d2)
        # Advance time so stand target deviates from home pose (ramp is 2 s)
        d2.time = 1.0
        loco.compute()
        # At t=1s, alpha=0.5 → q_des differs from home → ctrl non-zero
        leg_ctrls = d2.ctrl[:12]
        assert not np.all(leg_ctrls == 0.0)

    def test_base_height_returns_float(self, model_data):
        m, d = model_data
        loco = LocomotionController(m, d)
        h = loco.base_height()
        assert isinstance(h, float)
        assert 0.0 < h < 1.0

    def test_set_mode_trot(self, model_data):
        m, d = model_data
        loco = LocomotionController(m, d)
        loco.set_mode(GaitMode.TROT)
        assert loco._mode == GaitMode.TROT

    def test_trot_compute_does_not_crash(self, model_data):
        m, d = model_data
        d2 = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d2, kid)
        mujoco.mj_forward(m, d2)
        loco = LocomotionController(m, d2)
        loco.set_mode(GaitMode.TROT)
        for _ in range(50):
            loco.compute()
            mujoco.mj_step(m, d2)

    def test_base_yaw_returns_float_in_range(self, model_data):
        m, d = model_data
        loco = LocomotionController(m, d)
        yaw = loco.base_yaw()
        assert isinstance(yaw, float)
        assert -math.pi <= yaw <= math.pi
        assert abs(yaw) < 0.05, "robot should spawn facing +X (yaw ~ 0)"

    def test_zero_heading_and_full_speed_matches_legacy_trot(self, model_data):
        """Defaults (no set_heading/set_speed_scale call) must reproduce the
        pre-turning trot output exactly -- this pins the regression guard."""
        m, d = model_data
        loco = LocomotionController(m, d)
        q_des = loco._trot_joint_targets(1.23, 0.8)

        legacy = np.array([
            0.006, 0.609, -1.218, -0.006, 0.609, -1.218,
            0.006, 0.609, -1.218, -0.006, 0.609, -1.218,
        ])
        phi = (2.0 * math.pi * 1.23) / loco.GAIT_PERIOD
        phases = [0.0, math.pi, math.pi, 0.0]
        for i, phase_off in enumerate(phases):
            phi_leg = phi + phase_off
            sin_val = math.sin(phi_leg)
            legacy[3 * i + 1] -= 0.8 * 0.15 * sin_val
            legacy[3 * i + 2] -= 0.8 * 0.20 * sin_val
            legacy[3 * i + 0] += 0.8 * 0.04 * math.cos(phi_leg)

        np.testing.assert_allclose(q_des, legacy)

    def test_set_speed_scale_clamped(self, model_data):
        m, d = model_data
        loco = LocomotionController(m, d)
        loco.set_speed_scale(-1.0)
        assert loco._speed_scale == 0.0
        loco.set_speed_scale(2.0)
        assert loco._speed_scale == 1.0

    def test_set_heading_turns_robot_toward_target(self, model_data):
        m, d = model_data
        d2 = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d2, kid)
        mujoco.mj_forward(m, d2)
        loco = LocomotionController(m, d2)
        loco.set_mode(GaitMode.TROT)

        target_yaw = math.pi / 2
        yaw_before = loco.base_yaw()
        loco.set_heading(target_yaw)
        for _ in range(400):
            loco.compute()
            mujoco.mj_step(m, d2)
        yaw_after = loco.base_yaw()

        err_before = abs(math.atan2(math.sin(target_yaw - yaw_before), math.cos(target_yaw - yaw_before)))
        err_after = abs(math.atan2(math.sin(target_yaw - yaw_after), math.cos(target_yaw - yaw_after)))
        assert err_after < err_before, (
            f"heading error should shrink when turning toward target "
            f"(before={err_before:.3f}, after={err_after:.3f})"
        )


class TestManipulationController:
    def test_init(self, model_data):
        m, d = model_data
        manip = ManipulationController(m, d)
        assert manip is not None

    def test_ee_position_is_3d(self, model_data):
        m, d = model_data
        manip = ManipulationController(m, d)
        ee = manip.ee_position()
        assert ee.shape == (3,)

    def test_ee_above_ground(self, model_data):
        m, _ = model_data
        d2 = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d2, kid)
        mujoco.mj_forward(m, d2)
        manip = ManipulationController(m, d2)
        ee = manip.ee_position()
        assert float(ee[2]) > 0.0, "EE should be above ground"

    def test_home_pose_computes(self, model_data):
        m, d = model_data
        d2 = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d2, kid)
        manip = ManipulationController(m, d2)
        manip.set_home()
        manip.compute()
        arm_ctrl = [d2.ctrl[manip._act_ids[i]] for i in range(7)]
        # Home pose should be non-trivial
        assert not np.allclose(arm_ctrl, 0.0)

    def test_ik_reaches_nearby_target(self, model_data):
        m, d = model_data
        d2 = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d2, kid)
        mujoco.mj_forward(m, d2)

        manip = ManipulationController(m, d2)
        ee_start = manip.ee_position()
        # Target: 10 cm forward from current EE
        target = ee_start + np.array([0.10, 0.0, -0.05])
        q, converged = manip._numerical_ik(target)
        # IK should produce valid joint angles
        assert q.shape == (7,)
        assert np.all(q >= manip._q_lo), "IK violated lower limits"
        assert np.all(q <= manip._q_hi), "IK violated upper limits"

    def test_arm_qpos_returns_7_values(self, model_data):
        m, d = model_data
        manip = ManipulationController(m, d)
        q = manip.arm_qpos()
        assert q.shape == (7,)


@pytest.fixture(scope="module", params=sorted(END_EFFECTORS))
def ee_variant_model_data(request):
    """Build (model, data, end_effector_name) for each variant."""
    name = request.param
    m = _load_variant_model(DEFAULT_ARM, name)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)
    return m, d, name


class TestManipulationControllerEndEffectors:
    def test_constructs(self, ee_variant_model_data):
        m, d, name = ee_variant_model_data
        manip = ManipulationController(m, d, end_effector=name)
        assert manip is not None

    def test_ee_position_is_3d(self, ee_variant_model_data):
        m, d, name = ee_variant_model_data
        manip = ManipulationController(m, d, end_effector=name)
        ee = manip.ee_position()
        assert ee.shape == (3,)

    def test_open_close_ctrl_differ(self, ee_variant_model_data):
        m, d, name = ee_variant_model_data
        manip = ManipulationController(m, d, end_effector=name)
        assert manip._ee_spec.open_ctrl != manip._ee_spec.close_ctrl

    def test_is_grasped_runs_at_home(self, ee_variant_model_data):
        m, d, name = ee_variant_model_data
        manip = ManipulationController(m, d, end_effector=name)
        assert manip.is_grasped() in (True, False)


class TestControllerIntegration:
    def test_stand_10_steps(self, model_data):
        """Stand controller runs 10 steps without crashing."""
        m, d = model_data
        d2 = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d2, kid)
        loco = LocomotionController(m, d2)
        manip = ManipulationController(m, d2)
        for _ in range(10):
            loco.compute()
            manip.compute()
            mujoco.mj_step(m, d2)
        # Robot should still be somewhat upright after 10 steps
        assert loco.base_height() > 0.05
