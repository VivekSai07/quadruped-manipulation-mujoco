"""Integration test: run the full loco-manipulation task headlessly."""
from __future__ import annotations

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
