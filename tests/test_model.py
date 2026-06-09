"""Tests for the combined MJCF model loading and properties."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


@pytest.fixture(scope="module")
def data(model) -> mujoco.MjData:
    return mujoco.MjData(model)


class TestModelLoad:
    def test_loads_without_error(self, model):
        assert model is not None

    def test_nq_at_least_28(self, model):
        # 7 (freejoint) + 12 (legs) + 7 (arm) + 2 (fingers) = 28
        # + 7 (cube freejoint) = 35
        assert model.nq >= 28

    def test_nu_equals_20(self, model):
        # 12 leg motors + 7 arm actuators + 1 gripper
        assert model.nu == 20

    def test_combined_mass_plausible(self, model):
        total = float(sum(model.body_mass))
        # Go2 ~11.5 kg + Panda×0.35 ~6.5 kg + cube 0.2 kg ≈ 18-25 kg
        assert 15.0 < total < 30.0, f"Unexpected total mass: {total:.2f} kg"

    def test_sensor_count(self, model):
        # 12 pos + 12 vel + 12 torque + 5 IMU + 2 ee + 1 cube = 44 minimum
        assert model.nsensordata >= 44


class TestBodyStructure:
    def test_base_link_exists(self, model):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        assert bid >= 0

    def test_panda_link0_exists(self, model):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_link0")
        assert bid >= 0

    def test_panda_link0_is_child_of_base_link(self, model):
        bl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        p0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_link0")
        assert model.body_parentid[p0_id] == bl_id

    def test_ee_site_exists(self, model):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        assert sid >= 0

    def test_target_cube_exists(self, model):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        assert bid >= 0

    def test_worktable_exists(self, model):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worktable")
        assert bid >= 0

    def test_target_plate_exists(self, model):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_plate")
        assert bid >= 0

    def test_cube_joint_exists(self, model):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        assert jid >= 0

    def test_cube_on_table_in_keyframe(self, model):
        """Cube z in keyframe must be above table surface (0.30 m)."""
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        qadr = int(model.jnt_qposadr[jid])
        cube_z = float(model.key_qpos[kid, qadr + 2])
        assert cube_z > 0.30, f"Cube z={cube_z:.3f} not above table surface (0.30 m)"

    def test_all_leg_joints_exist(self, model):
        leg_joints = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        for name in leg_joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint not found: {name}"

    def test_all_panda_joints_exist(self, model):
        for i in range(1, 8):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            assert jid >= 0, f"joint{i} not found"

    def test_home_keyframe_exists(self, model):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert kid >= 0


class TestKeyframeReset:
    def test_reset_to_home_does_not_crash(self, model, data):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, kid)
        mujoco.mj_forward(model, data)

    def test_base_height_after_reset(self, model, data):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, kid)
        mujoco.mj_forward(model, data)
        # After reset to home keyframe, base should be at ~0.27 m
        base_z = float(data.qpos[2])
        assert 0.1 < base_z < 0.5, f"Unexpected base height: {base_z:.3f} m"

    def test_simulation_step_does_not_crash(self, model, data):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, kid)
        for _ in range(10):
            mujoco.mj_step(model, data)
