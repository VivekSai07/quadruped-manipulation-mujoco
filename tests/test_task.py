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
from tasks.pick_and_place import PickAndPlaceTask
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
            TaskState.ADJUSTING_HEIGHT, TaskState.MANIPULATING,
            TaskState.RETURNING_HOME, TaskState.DONE,
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


class TestSeedApproachSeedsPhaseEnterTime:
    """Regression test for the MANIPULATING-entry bug: `seed_approach()` used
    to only seed `_arm_interp_target`, leaving `_phase_enter_time` at its
    `__init__` default of 0.0 (APPROACHING is never reached via `_set_phase()`
    -- it's the initial phase). On the very first `manip_step` call, `elapsed
    = t - self._phase_enter_time` then evaluated to the absolute sim time `t`
    instead of true time-in-phase, silently bypassing the `min_approach_time`
    floor whenever the EE happened to already be within `approach_threshold`
    of the approach waypoint at MANIPULATING-entry time.

    This test constructs exactly that scenario: the cube/target are placed so
    that the approach waypoint coincides with the arm's home-pose EE position
    -- i.e. `ee_distance_to(wp_approach) == 0 < approach_threshold` from the
    very first `manip_step` call, with no IK convergence needed. If
    `seed_approach` does not seed `_phase_enter_time = t`, the very first
    `manip_step` call incorrectly transitions APPROACHING -> DESCENDING
    immediately (confirmed by manually reproducing the pre-fix code path)."""

    def _build_task_with_ee_at_approach_waypoint(
        self, cfg
    ) -> tuple[PickAndPlaceTask, mujoco.MjModel, mujoco.MjData, "ManipulationController"]:
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)

        from controllers.manipulation import ManipulationController  # noqa: PLC0415

        manip = ManipulationController(m, d)
        ee_home = manip.ee_position().copy()

        # Place the cube directly below the home-pose EE position by the
        # hover height (_HOVER_Z=0.15) so wp_approach == cz+hover == ee_home,
        # with zero IK travel needed -- dist is exactly 0 at construction time.
        hover_z = 0.15
        ftp_offset = manip._ee_spec.ftp_offset
        cube_pos = [float(ee_home[0]), float(ee_home[1]), float(ee_home[2] - hover_z)]
        target_pos = [float(ee_home[0]) + 0.3, float(ee_home[1]), float(ee_home[2] - hover_z) + 0.006]

        task_cfg = dict(cfg["task"])
        task = PickAndPlaceTask(cube_pos, target_pos, m, d, manip, ftp_offset, task_cfg)
        return task, m, d, manip

    def test_seed_approach_sets_phase_enter_time_to_t(self, cfg):
        """seed_approach(t) must seed _phase_enter_time, not leave it at 0.0."""
        task, _m, _d, _manip = self._build_task_with_ee_at_approach_waypoint(cfg)
        assert task._phase_enter_time == 0.0  # __init__ default, pre-seed

        late_t = 42.0
        task.seed_approach(late_t)
        assert task._phase_enter_time == late_t

    def test_min_approach_time_floor_holds_when_ee_starts_at_waypoint(self, cfg):
        """Even when the EE is already exactly at the approach waypoint at
        MANIPULATING-entry time (dist=0 < approach_threshold), the task must
        not transition out of APPROACHING before min_approach_time has
        genuinely elapsed since entry. Before the fix this transitioned on
        the very first manip_step call regardless of min_approach_time,
        because `elapsed` was computed against absolute sim time (t - 0.0)
        instead of time-since-entry."""
        task, m, _d, manip = self._build_task_with_ee_at_approach_waypoint(cfg)
        assert manip.ee_distance_to(task._wp_approach) < task._approach_threshold

        entry_t = 50.0  # arbitrary late, non-zero absolute sim time
        task.seed_approach(entry_t)

        dt = m.opt.timestep
        # Step for well under min_approach_time and confirm we have NOT left
        # APPROACHING -- this is the floor the bug silently bypassed.
        steps_before_floor = max(1, int((task._min_approach_time * 0.5) / dt))
        t = entry_t
        for _ in range(steps_before_floor):
            t += dt
            task.manip_step(coordinator=None, t=t, dt=dt)

        assert task.phase == PickAndPlaceTask.Phase.APPROACHING, (
            "APPROACHING ended before min_approach_time elapsed since entry -- "
            "_phase_enter_time was not seeded correctly by seed_approach()"
        )

        # And once min_approach_time genuinely has elapsed (with the EE still
        # at the waypoint), the transition to DESCENDING does occur --
        # confirms the floor is a real gate, not something permanently stuck.
        for _ in range(steps_before_floor + 5):
            t += dt
            task.manip_step(coordinator=None, t=t, dt=dt)
        assert task.phase == PickAndPlaceTask.Phase.DESCENDING
