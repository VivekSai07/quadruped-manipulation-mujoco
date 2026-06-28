"""
High-level task coordinator for the loco-manipulation demo.

State machine:
  INIT -> STANDING -> WALKING -> STOPPING -> STABILIZING
       -> ADJUSTING_HEIGHT                          (adaptive Go2 crouch; skipped if delta=0)
       -> MANIPULATING                               (delegates to the active Task)
       -> RETURNING_HOME -> DONE

Key design decisions
--------------------
* Generic outer machine: locomotion (WALKING), stabilization, and adaptive
  height all live here, independent of what manipulation task is running.
  The MANIPULATING state simply calls `self._active_task.manip_step(...)`
  every step; the task's own internal sub-states (e.g. approach/descend/
  grasp/lift/transport/lower/release for pick-and-place) live inside the
  concrete Task subclass, invisible to this coordinator.

* Velocity IK (archive m02 pattern): manipulation tasks integrate one
  Jacobian step per physics timestep instead of firing a batch IK every 0.3 s.
  This eliminates the 3 cm position lurches that caused jerky arm motion.

* Adaptive height (ADJUSTING_HEIGHT): after stabilizing, workspace analysis
  computes a crouch alpha from (a) horizontal arm extension and (b) vertical
  arm extension below base. The locomotion controller blends stand/crouch poses
  without any hardcoded height number --the decision is scenario-driven.

* RETURNING_HOME: after the active task finishes, the arm joint-interpolates
  back to the home pose before the simulation ends (or the next task begins).
"""
from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from .arms import DEFAULT_ARM
from .end_effectors import DEFAULT_END_EFFECTOR
from .locomotion import GaitMode, LocomotionController
from .manipulation import ManipulationController

if TYPE_CHECKING:
    from tasks.base import Task


class TaskState(enum.Enum):
    INIT             = "init"
    STANDING         = "standing"
    WALKING          = "walking"
    STOPPING         = "stopping"
    STABILIZING      = "stabilizing"       # pause after stopping, arm at home
    ADJUSTING_HEIGHT = "adjusting_height"  # Go2 adaptive crouch for arm workspace
    MANIPULATING     = "manipulating"      # delegated to the active Task
    RETURNING_HOME   = "returning_home"    # arm returns to home pose
    DONE             = "done"


_MIN_STAND_HEIGHT   = 0.22    # m --robot considered stably upright above this
_STOP_VEL_THRESHOLD = 0.06   # m/s --base XY speed to consider "stopped"

# Joint-space return rate for RETURNING_HOME (rad/s in L2 joint space).
# 1.5 rad/s means 1.1 rad total error reaches home in ~0.75 s of commanded travel.
_JOINT_RETURN_RATE = 1.5  # rad/s


class TaskCoordinator:
    """Orchestrates locomotion and the active manipulation Task."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: dict[str, Any],
        arm: str = DEFAULT_ARM,
        end_effector: str = DEFAULT_END_EFFECTOR,
        task: "Task | None" = None,
    ) -> None:
        self.model = model
        self.data  = data
        self._cfg  = config

        task_cfg = config.get("task", {})

        # Sub-controllers
        self.loco  = LocomotionController(model, data)
        self.manip = ManipulationController(model, data, arm=arm, end_effector=end_effector)
        self._ftp  = self.manip._ee_spec.ftp_offset

        # State machine
        self._state            = TaskState.INIT
        self._stable_since:  float = -999.0
        self._stop_since:    float = -999.0
        self._state_enter_time: float = 0.0

        # Generic task config (governs WALKING/STABILIZING/ADJUSTING_HEIGHT)
        self._stop_distance       = task_cfg.get("stop_distance",       0.65)
        self._stable_duration     = task_cfg.get("stable_duration",      2.0)
        self._stabilize_duration  = task_cfg.get("stabilize_duration",   2.0)
        self._height_settle_time  = task_cfg.get("height_settle_time",   2.0)

        # Active task: build the default PickAndPlaceTask when none is given,
        # exactly as today --this is what keeps all existing call sites working
        # unmodified.
        if task is None:
            from tasks.pick_and_place import PickAndPlaceTask  # noqa: PLC0415

            cube_pos   = task_cfg.get("cube_pos",   [1.6, 0.0,  0.325])
            target_pos = task_cfg.get("target_pos", [1.6, 0.20, 0.331])
            task = PickAndPlaceTask(
                cube_pos, target_pos, model, data, self.manip, self._ftp, task_cfg,
            )
        self._active_task = task

        # Commanded joint target for RETURNING_HOME interpolation.
        # Advanced at _JOINT_RETURN_RATE per step independently of measured qpos,
        # so the target races to home quickly regardless of PD tracking lag.
        self._return_q_target: np.ndarray = self.manip.home_qpos()

    # ── Adaptive height analysis ───────────────────────────────────────────

    def _compute_height_adjustment(self, descend_point: np.ndarray) -> float:
        """Return crouch alpha (0-1) for the current reach scenario.

        Decision factors:
          - Horizontal reach: arm extends forward to the target from the
            robot's stopped position. Longer reach -> more forward torque on
            Go2 body -> lower CoM improves balance.
          - Vertical reach below base: arm descends below Go2's base height.
            More downward reach -> lowering base reduces arm extension.

        Both factors are normalised and summed; the result is clamped to [0, 0.5]
        so Go2 never goes beyond a safe ~3 cm lower stance.
        """
        base_z  = float(self.data.qpos[2])
        robot_x = float(self.data.qpos[0])

        horizontal_reach = abs(descend_point[0] - robot_x)
        vertical_reach   = max(0.0, base_z - descend_point[2])

        # Contribution weights --tuned so current scenario gives ~0.25-0.35
        alpha_h = max(0.0, (horizontal_reach - 0.30) / 0.60)  # 0 at 30 cm, 1 at 90 cm
        alpha_v = max(0.0, (vertical_reach   - 0.00) / 0.30)  # 0 at 0 cm below, 1 at 30 cm

        alpha = min(0.5, alpha_h * 0.35 + alpha_v * 0.25)
        return alpha

    # ── Main step ─────────────────────────────────────────────────────────

    def step(self, t: float, dt: float) -> None:
        """Call once per simulation timestep."""
        self._update_state(t, dt)
        self.loco.compute()
        self.manip.compute()

    # ── State machine ─────────────────────────────────────────────────────

    def _update_state(self, t: float, dt: float) -> None:  # noqa: PLR0912, PLR0915
        state = self._state

        if state == TaskState.INIT:
            self._transition(TaskState.STANDING, t)

        elif state == TaskState.STANDING:
            height = self.loco.base_height()
            if height > _MIN_STAND_HEIGHT:
                if self._stable_since < 0:
                    self._stable_since = t
                elif t - self._stable_since >= self._stable_duration:
                    self.loco.set_mode(GaitMode.TROT)
                    self._transition(TaskState.WALKING, t)
            else:
                self._stable_since = -999.0

        elif state == TaskState.WALKING:
            target_xy = self._active_task.target_xy()
            dist = self._base_xy_distance_to_target()

            base_xy = self.loco.base_position()[:2]
            bearing = math.atan2(target_xy[1] - base_xy[1], target_xy[0] - base_xy[0])
            self.loco.set_heading(bearing)

            # Slow down only in the final approach, not across the whole walk --
            # stop_distance is already a wide clearance buffer (0.65 m default),
            # so ramping over 2x that would dominate total walk time for a short
            # walk. A fixed 0.3 m final-approach margin keeps the ramp local to
            # the actual stop point regardless of total distance walked.
            slow_zone = self._stop_distance + 0.3
            if dist < slow_zone:
                frac = max(0.0, (dist - self._stop_distance) / (slow_zone - self._stop_distance))
                self.loco.set_speed_scale(0.5 + 0.5 * frac)  # floor at 0.5, never fully stall
            else:
                self.loco.set_speed_scale(1.0)

            if dist < self._stop_distance:
                self.loco.set_mode(GaitMode.STAND)
                self.loco.clear_heading()
                self._stop_since = t
                self._transition(TaskState.STOPPING, t)

        elif state == TaskState.STOPPING:
            vel     = float(np.linalg.norm(self.loco.base_velocity()[:2]))
            elapsed = t - self._stop_since
            if vel < _STOP_VEL_THRESHOLD or elapsed > 3.0:
                self.manip.set_home()
                self.manip.set_gripper(open_=True)
                self._transition(TaskState.STABILIZING, t)

        elif state == TaskState.STABILIZING:
            if t - self._state_enter_time >= self._stabilize_duration:
                self.manip.set_gripper(open_=True)
                # Compute scenario-aware crouch before arm starts moving.
                # This path is specific to PickAndPlaceTask's descend point
                # (default-constructed task path); see approach_descend_point().
                descend_point = self._active_task.approach_descend_point()
                alpha = self._compute_height_adjustment(descend_point)
                if alpha > 0.02:
                    self.loco.set_crouch_alpha(alpha)
                    print(
                        f"  [t={t:.2f}s] Adaptive height: crouch_alpha={alpha:.2f} "
                        f"(horizontal={abs(descend_point[0]-self.data.qpos[0]):.2f}m "
                        f"below_base={max(0,self.data.qpos[2]-descend_point[2]):.2f}m)"
                    )
                    self._transition(TaskState.ADJUSTING_HEIGHT, t)
                else:
                    # No meaningful height adjustment --proceed directly
                    self._active_task.seed_approach(t)
                    self._transition(TaskState.MANIPULATING, t)

        elif state == TaskState.ADJUSTING_HEIGHT:
            # Wait for the new stance to settle
            if t - self._state_enter_time >= self._height_settle_time:
                self._active_task.seed_approach(t)
                self._transition(TaskState.MANIPULATING, t)

        elif state == TaskState.MANIPULATING:
            done = self._active_task.manip_step(self, t, dt)
            if done:
                self._active_task.is_success()
                # Seed return target from current MEASURED arm pose so the
                # commanded trajectory starts exactly where the arm is now
                self._return_q_target = self.manip.arm_qpos().copy()
                self._transition(TaskState.RETURNING_HOME, t)

        elif state == TaskState.RETURNING_HOME:
            # Advance the commanded joint target toward home at _JOINT_RETURN_RATE.
            # Working from the COMMANDED target (not measured qpos) means the target
            # races to home at full speed regardless of PD tracking lag.
            q_home = self.manip.home_qpos()
            delta  = q_home - self._return_q_target
            dist   = float(np.linalg.norm(delta))
            step   = _JOINT_RETURN_RATE * dt
            if dist <= step:
                self._return_q_target = q_home.copy()
                self.manip.set_joint_target(q_home)
                next_task = self._active_task.next_task()
                if next_task is not None:
                    self._active_task = next_task
                    self._active_task.seed_approach(t)
                    # Re-entering WALKING here skips STANDING, which is the
                    # only other place gait mode is set to TROT -- without
                    # this, the locomotion controller stays in whatever mode
                    # WALKING last left it (STAND, set on arrival at the
                    # previous task's stop point) and the robot never
                    # actually walks toward the next task's target.
                    self.loco.set_mode(GaitMode.TROT)
                    self._transition(TaskState.WALKING, t)
                else:
                    self._transition(TaskState.DONE, t)
            else:
                self._return_q_target += delta / dist * step
                self.manip.set_joint_target(self._return_q_target)

        # DONE: hold last position

    # ── Helpers ───────────────────────────────────────────────────────────

    def _transition(self, new_state: TaskState, t: float) -> None:
        print(f"  [t={t:.2f}s] {self._state.value} -> {new_state.value}")
        self._state            = new_state
        self._state_enter_time = t

    def _base_xy_distance_to_target(self) -> float:
        return float(np.linalg.norm(self.loco.base_position()[:2] - self._active_task.target_xy()))

    def post_physics_step(self) -> None:
        """Re-enforce any task-specific post-step physics fixup (e.g. the
        pick-and-place 6-DOF kinematic cube attachment), if the active task
        defines one. Not every Task subclass needs this, so this stays a
        conditional dispatch rather than a new abstract method on Task --
        the coordinator's own public API must keep working unconditionally
        for every task type, including future ones that never define it.
        """
        if hasattr(self._active_task, "post_physics_step"):
            self._active_task.post_physics_step()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def is_done(self) -> bool:
        return self._state == TaskState.DONE

    @property
    def active_task(self) -> "Task":
        return self._active_task

    def status_line(self, t: float) -> str:
        h        = self.loco.base_height()
        dist_base = self._base_xy_distance_to_target()
        grasped   = self.manip.is_grasped()
        state     = self._state

        if state == TaskState.MANIPULATING:
            ee_dist = 0.0
            lbl     = "manipulating"
        elif state == TaskState.RETURNING_HOME:
            ee_dist   = float(np.linalg.norm(
                self.manip.arm_qpos() - self.manip.home_qpos()))
            lbl       = "q_err(home)"
        else:
            ee_dist = 0.0
            lbl     = "manipulating"

        return (
            f"t={t:5.2f}s | {state.value:18s} | "
            f"h={h:.3f}m | base->target={dist_base:.3f}m | "
            f"{lbl}={ee_dist:.3f}m | grasped={grasped}"
        )
