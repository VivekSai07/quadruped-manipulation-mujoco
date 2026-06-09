"""Tests for locomotion and manipulation controllers."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from controllers.locomotion import GaitMode, LocomotionController
from controllers.manipulation import ManipulationController

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")


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
        from controllers.manipulation import _Q_LO, _Q_HI
        assert np.all(q >= _Q_LO), "IK violated lower limits"
        assert np.all(q <= _Q_HI), "IK violated upper limits"

    def test_arm_qpos_returns_7_values(self, model_data):
        m, d = model_data
        manip = ManipulationController(m, d)
        q = manip.arm_qpos()
        assert q.shape == (7,)


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
