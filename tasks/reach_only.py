"""
Reach-only task: Go2 walks toward a fixed point, Panda (or swapped arm)
reaches the end-effector to it. No grasp/lift/transport/lower -- this is
intentionally "just the approach phase, nothing after," to prove the Task
abstraction (tasks/base.py) generalizes beyond pick-and-place.

Reuses the same velocity-IK interpolation pattern as PickAndPlaceTask's
APPROACHING phase (controllers/manipulation.py's reach_position_smooth),
and the same approach_threshold/min_approach_time config keys -- no new
config schema is introduced here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from .base import Task

if TYPE_CHECKING:
    from controllers.coordinator import TaskCoordinator
    from controllers.manipulation import ManipulationController

# Cartesian target move rate for the single reaching phase (m/s) -- matches
# PickAndPlaceTask's APPROACHING rate (_ARM_MOVE_RATE in tasks/pick_and_place.py).
_ARM_MOVE_RATE = 0.10  # 10 cm/s


class ReachOnlyTask(Task):
    """Walk-to-point, reach-only task: no grasp, lift, transport, or lower.

    Has exactly one internal phase (reaching) -- no Phase enum needed."""

    def __init__(
        self,
        target_point: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        manip: "ManipulationController",
        task_cfg: dict[str, Any],
    ) -> None:
        self.model = model
        self.data = data
        self.manip = manip

        self._target_point = np.array(target_point, dtype=np.float64)

        self._approach_threshold = task_cfg.get("approach_threshold", 0.05)
        self._min_approach_time = task_cfg.get("min_approach_time", 1.5)

        # Smooth Cartesian interpolation target (updated each step) -- seeded
        # for real by seed_approach(); this is just a safe pre-seed default.
        self._arm_interp_target: np.ndarray = np.zeros(3)

        # Single-phase entry timestamp (mirrors PickAndPlaceTask's
        # _phase_enter_time, but there's only one phase here).
        self._phase_enter_time: float = 0.0

    # ── Smooth interp target ─────────────────────────────────────────────

    def _step_interp_target(self, goal: np.ndarray, dt: float) -> np.ndarray:
        """Advance self._arm_interp_target one step toward goal at
        _ARM_MOVE_RATE, mirroring PickAndPlaceTask._step_interp_target."""
        step = _ARM_MOVE_RATE * dt
        delta = goal - self._arm_interp_target
        dist = float(np.linalg.norm(delta))
        if dist <= step:
            self._arm_interp_target = goal.copy()
        else:
            self._arm_interp_target += delta / dist * step
        return self._arm_interp_target

    def seed_approach(self, t: float) -> None:
        """Seed the interpolated arm target at the current EE position (no
        jerk on state entry) and seed the phase-entry timestamp to t.

        Required unconditionally by TaskCoordinator on MANIPULATING entry
        (controllers/coordinator.py) even though it is not part of the
        formal Task ABC -- see tasks/pick_and_place.py's seed_approach for
        the original rationale (without seeding _phase_enter_time, the
        first manip_step's `elapsed = t - phase_enter_time` would compute
        against absolute sim time instead of true time-in-phase)."""
        self._arm_interp_target = self.manip.ee_position().copy()
        self._phase_enter_time = t

    def approach_descend_point(self) -> np.ndarray:
        """Return the point the arm reaches toward, for the coordinator's
        generic height-adjustment math (_compute_height_adjustment). For a
        reach-only task there is no separate hover/descend distinction --
        the target point itself is the descend point."""
        return self._target_point

    # ── Task ABC interface ────────────────────────────────────────────────

    def target_xy(self) -> np.ndarray:
        return self._target_point[:2]

    def manip_step(self, coordinator: "TaskCoordinator", t: float, dt: float) -> bool:
        """Drive the EE toward the single target point. Returns True once
        within threshold (and min_approach_time has genuinely elapsed) --
        there is no further sub-state machine."""
        current_target = self._step_interp_target(self._target_point, dt)
        self.manip.reach_position_smooth(current_target, dt)
        elapsed = t - self._phase_enter_time
        dist = self.manip.ee_distance_to(self._target_point)
        return elapsed >= self._min_approach_time and dist < self._approach_threshold

    def is_success(self) -> bool:
        return self.manip.ee_distance_to(self._target_point) < self._approach_threshold
