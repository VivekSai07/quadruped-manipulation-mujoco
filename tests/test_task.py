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


class TestRegraspFaultRecovery:
    """Task 3: GRASPING's failure path must retry via REGRASP up to
    `max_grasp_attempts` before giving up, instead of silently proceeding
    ungrasped on the very first missed contact.

    `ManipulationController.is_grasped()` (controllers/manipulation.py) is
    purely contact-physics-based (checks `data.contact` for both fingers
    touching the cube body), so it cannot be forced false/true by adjusting
    waypoints without also fighting real contact dynamics. We monkeypatch it
    directly -- a deterministic, unit-level way to drive the GRASPING/REGRASP
    branches without depending on contact solver timing.
    """

    def _build_task_at_grasping(self, cfg):
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)

        from controllers.manipulation import ManipulationController  # noqa: PLC0415

        manip = ManipulationController(m, d)
        ee_home = manip.ee_position().copy()

        hover_z = 0.15
        ftp_offset = manip._ee_spec.ftp_offset
        cube_pos = [float(ee_home[0]), float(ee_home[1]), float(ee_home[2] - hover_z)]
        # +0.1m in X (not +0.3, see TestSeedApproachSeedsPhaseEnterTime's variant
        # -- that test never drives past DESCENDING, so its larger offset never
        # gets exercised through TRANSPORTING/LOWERING). +0.3m places
        # _wp_transport/_wp_lower outside the Panda's reachable workspace from
        # this arm's home pose -- the velocity IK never converges and the EE
        # stalls in TRANSPORTING forever (confirmed via direct IK probing: a
        # batch `reach_position` call to that point does not converge either).
        # +0.1m is confirmed reachable (IK converges within ~2mm for both
        # _wp_transport and _wp_lower at this hover height).
        target_pos = [float(ee_home[0]) + 0.1, float(ee_home[1]), float(ee_home[2] - hover_z) + 0.006]

        # Relocate the *real* physics cube (it spawns far away at the model's
        # hardcoded [1.6, 0, 0.325], same caveat as TestWalkingTurnsTowardOff
        # AxisCube) to sit right under the EE, matching `cube_pos` above. The
        # 6-DOF kinematic attachment (_apply_kinematic_attachment) computes
        # `_grasp_offset = cube_pos - ee_pos` at grasp-confirm time and then
        # teleports the cube to `ee_pos + offset` every LIFTING/TRANSPORTING/
        # LOWERING step -- with the real cube left at its 1m+ away spawn
        # point, that offset is huge and the teleported cube ends up
        # clipping through the robot's own body, generating contact forces
        # that fight the arm's actuators and stall TRANSPORTING forever
        # (discovered while debugging this test never reaching done).
        cube_jid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qadr = int(m.jnt_qposadr[cube_jid])
        d.qpos[cube_qadr:cube_qadr + 3] = cube_pos
        mujoco.mj_forward(m, d)

        task_cfg = dict(cfg["task"])
        task = PickAndPlaceTask(cube_pos, target_pos, m, d, manip, ftp_offset, task_cfg)

        # Jump straight to GRASPING, holding at the descend waypoint, as if
        # DESCENDING had just converged -- this test is about the GRASPING/
        # REGRASP retry logic, not the earlier approach/descend phases.
        entry_t = 10.0
        task._arm_interp_target = task._wp_descend.copy()
        task._set_phase(PickAndPlaceTask.Phase.GRASPING, entry_t)
        return task, m, d, manip, entry_t

    @staticmethod
    def _advance(task, manip, m, d, t, dt) -> tuple[float, bool]:
        """One full physics tick, mirroring TaskCoordinator.step() + the
        run_simulation.py main loop: manip_step() (the sub-state machine),
        then manip.compute() (writes the IK-commanded qpos into ctrl), then
        mj_step() (actually integrates the PD-controlled actuators), then
        post_physics_step() (re-applies the kinematic cube lock).

        `reach_position_smooth()` only ever updates `manip._q_target` and
        leaves `data.qpos` untouched -- physical EE motion only happens once
        `compute()` + `mj_step()` run. Calling `manip_step()` in a tight loop
        without this would freeze the EE in place forever (discovered while
        debugging an infinite REGRASP loop in this test)."""
        t += dt
        done = task.manip_step(coordinator=None, t=t, dt=dt)
        manip.compute()
        mujoco.mj_step(m, d)
        task.post_physics_step()
        return t, done

    def _run_until(self, task, manip, m, d, t0, dt, max_steps=20000, predicate=None):
        """Advance physics forward until predicate(task) is True or manip_step
        returns True (done), whichever comes first. Returns (t, done)."""
        t = t0
        for _ in range(max_steps):
            t, done = self._advance(task, manip, m, d, t, dt)
            if done:
                return t, True
            if predicate is not None and predicate(task):
                return t, False
        raise AssertionError("manip_step did not satisfy predicate/finish within max_steps")

    def test_failed_grasp_retries_then_succeeds_and_reaches_done(self, cfg, capsys, monkeypatch):
        """First grasp evaluation fails (no contact) -> logs a retry line and
        enters REGRASP. Second grasp evaluation (post re-descend) succeeds ->
        task hands off into LIFTING with a confirmed grasp.

        Deliberately stops short of asserting full pipeline completion
        (RELEASING/DONE) -- see the comment below the REGRASP-entry
        assertions for why."""
        task, m, d, manip, entry_t = self._build_task_at_grasping(cfg)
        dt = m.opt.timestep

        grasp_calls = {"count": 0}

        def fake_is_grasped():
            grasp_calls["count"] += 1
            # Fail the first evaluation, succeed on the second (the retry).
            return grasp_calls["count"] >= 2

        monkeypatch.setattr(manip, "is_grasped", fake_is_grasped)

        # Run until REGRASP is entered (first failed attempt).
        t, done = self._run_until(
            task, manip, m, d, entry_t, dt,
            predicate=lambda tk: tk.phase == PickAndPlaceTask.Phase.REGRASP,
        )
        assert not done
        assert task.phase == PickAndPlaceTask.Phase.REGRASP
        assert task._grasp_attempts == 1

        out = capsys.readouterr().out
        assert "Grasp attempt 1 failed" in out
        assert "retry 2/2" in out
        assert "Grasp confirmed" not in out
        assert "WARNING: no contact" not in out

        # Continue stepping: REGRASP re-descends, re-enters GRASPING, and the
        # second evaluation succeeds. We assert on the REGRASP/GRASPING
        # hand-off itself (phase reaches LIFTING with a confirmed grasp) and
        # deliberately stop short of asserting full pipeline completion
        # (RELEASING/DONE). LIFTING/TRANSPORTING's steady-state IK tracking
        # has a pre-existing gravity/PD-droop issue (confirmed to reproduce
        # even with the original pre-Task-3 single-failure-then-give-up
        # fallback, no REGRASP code involved) that can stall convergence once
        # the EE starts a non-trivial distance from _wp_transport -- which is
        # exactly what an ungrasped/regrasp-displaced LIFTING does. That bug
        # is out of this task's scope (LIFTING/TRANSPORTING are explicitly
        # off-limits per the brief) and unrelated to what Task 3 changed.
        # See .superpowers/sdd/task-3-report.md for the full root-cause
        # writeup and the decision to descope this assertion.
        t, done = self._run_until(
            task, manip, m, d, t, dt,
            predicate=lambda tk: tk.phase == PickAndPlaceTask.Phase.LIFTING,
        )
        assert not done
        assert task.phase == PickAndPlaceTask.Phase.LIFTING
        assert task._grasp_attempts == 2
        assert task._grasp_confirmed

        out2 = capsys.readouterr().out
        assert "Grasp confirmed" in out2
        assert "WARNING: no contact" not in out2

    def test_exhausted_retries_gives_up_and_proceeds_ungrasped(self, cfg, capsys, monkeypatch):
        """Grasp never succeeds. With default max_grasp_attempts=2, the task
        must retry exactly once via REGRASP, then give up (log the existing
        WARNING line) and proceed ungrasped into LIFTING -- never getting
        stuck in REGRASP forever.

        Stops short of asserting RELEASING/DONE for the same pre-existing,
        out-of-scope reason documented in the test above."""
        task, m, d, manip, entry_t = self._build_task_at_grasping(cfg)
        dt = m.opt.timestep

        monkeypatch.setattr(manip, "is_grasped", lambda: False)

        t, done = self._run_until(
            task, manip, m, d, entry_t, dt,
            predicate=lambda tk: tk.phase == PickAndPlaceTask.Phase.LIFTING,
        )
        assert not done
        assert task.phase == PickAndPlaceTask.Phase.LIFTING

        # Exactly one retry occurred before giving up: 2 grasp evaluations.
        assert task._grasp_attempts == 2
        assert not task._grasp_confirmed

        out = capsys.readouterr().out
        assert "Grasp attempt 1 failed" in out
        assert "retry 2/2" in out
        assert "WARNING: no contact --lifting without lock" in out
        assert "Grasp confirmed" not in out
