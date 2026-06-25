"""Integration test: run the full loco-manipulation task headlessly."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from controllers.coordinator import TaskCoordinator, TaskState  # noqa: F401
from tasks.reach_task import ReachTask

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")
CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))


class TestTaskCoordinator:
    def test_coordinator_init(self, cfg):
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        coord = TaskCoordinator(m, d, cfg)
        assert coord.state == TaskState.INIT

    def test_coordinator_transitions_to_standing(self, cfg):
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)
        coord = TaskCoordinator(m, d, cfg)
        coord.step(0.0, m.opt.timestep)
        assert coord.state == TaskState.STANDING

    def test_task_runs_5_seconds(self, cfg):
        """Task coordinator runs 5 simulated seconds without error."""
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        task = ReachTask(m, d, cfg)
        dt = m.opt.timestep
        n_steps = int(5.0 / dt)
        for _ in range(n_steps):
            task.step(dt)
            mujoco.mj_step(m, d)
        # Should reach at least STANDING state within 5 seconds
        assert task.coordinator.state in (
            TaskState.STANDING, TaskState.WALKING,
            TaskState.STOPPING, TaskState.STABILIZING,
            TaskState.APPROACHING, TaskState.DESCENDING,
            TaskState.GRASPING, TaskState.LIFTING,
            TaskState.TRANSPORTING, TaskState.LOWERING,
            TaskState.RELEASING, TaskState.DONE,
        )

    def test_robot_stays_upright_during_task(self, cfg):
        """Robot should not collapse during first 5 seconds of task."""
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        task = ReachTask(m, d, cfg)
        dt = m.opt.timestep
        n_steps = int(5.0 / dt)
        min_height = 999.0
        for _ in range(n_steps):
            task.step(dt)
            mujoco.mj_step(m, d)
            h = task.coordinator.loco.base_height()
            min_height = min(min_height, h)
        assert min_height > 0.10, f"Robot collapsed at height {min_height:.3f} m"


class TestWalkingTurnsTowardOffAxisCube:
    """The default demo cube is always straight ahead (configs/default.yaml),
    so nothing else exercises non-zero bearing. Verify WALKING actually steers
    toward an off-axis target instead of just walking straight.

    configs/default.yaml's task.cube_pos only seeds TaskCoordinator's initial
    waypoint math -- the cube's real physical position is hardcoded in the
    compiled model (scripts/build_model.py: <body name="target_cube" pos="1.6
    0 0.325">) and is NOT wired from config. TaskCoordinator._refresh_cube_pos()
    re-reads the real physics position every WALKING step, so a config-only
    override is silently ineffective. To genuinely exercise off-axis turning,
    the cube's freejoint qpos must be relocated directly before stepping."""

    def test_walking_state_turns_toward_offaxis_cube(self, cfg):
        offaxis_cfg = copy.deepcopy(cfg)
        offaxis_cfg["task"]["cube_pos"] = [1.0, 1.0, 0.325]
        offaxis_cfg["task"]["target_pos"] = [1.0, 1.2, 0.331]

        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)

        cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qadr = int(m.jnt_qposadr[cube_jid])
        d.qpos[cube_qadr:cube_qadr + 3] = offaxis_cfg["task"]["cube_pos"]
        mujoco.mj_forward(m, d)

        coord = TaskCoordinator(m, d, offaxis_cfg)

        dt = m.opt.timestep
        target_yaw = math.atan2(1.0, 1.0)
        yaw_before = coord.loco.base_yaw()
        n_steps = int(8.0 / dt)
        for _ in range(n_steps):
            coord.step(d.time, dt)
            mujoco.mj_step(m, d)
        yaw_after = coord.loco.base_yaw()

        def _err(yaw: float) -> float:
            diff = target_yaw - yaw
            return abs(math.atan2(math.sin(diff), math.cos(diff)))

        assert coord.state in (TaskState.WALKING, TaskState.STOPPING, TaskState.STABILIZING)
        assert _err(yaw_after) < _err(yaw_before), (
            "WALKING should steer base_yaw toward the off-axis cube's bearing "
            f"(target={target_yaw:.3f}, before={yaw_before:.3f}, after={yaw_after:.3f})"
        )
