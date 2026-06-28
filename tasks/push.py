"""
Push task: Go2 walks up to a pushable object, Panda (or swapped arm) drives
its end-effector through the object via real contact-friction physics --
no grasp, no kinematic attachment. This is the deliberately "less
deterministic" counterpart to pick-and-place: success depends on whatever
the contact solver actually does to the object, not a lock this task forces.

Three internal phases, not two -- see docs/plans/2026-06-26-task-framework-
generalization.md / .superpowers/sdd/task-5-brief.md for the full reasoning.
A literal "approach, then push in a straight line" reading doesn't work
physically: moving from "hovering above the object" straight to "at push
height above the target" in one 3D line descends and translates in XY at
the same time, so by the time the EE reaches push height it may have
already overshot the object's XY without ever making solid contact. To get
a real push, the EE must reach contact height *while still above the
object*, before it starts translating horizontally:

    APPROACHING  -- hover above the object (identical pattern to
                    PickAndPlaceTask.Phase.APPROACHING)
    DESCENDING   -- drop straight down (XY fixed at the object's XY) to
                    push/contact height
    PUSHING      -- translate horizontally at that fixed height from the
                    object's XY to the target's XY

The gripper closes only during PUSHING, and only to present a flat contact
surface -- never to grasp. An open gripper's fingers are spaced wider than
the pushable cube and straddle it with zero contact (confirmed empirically:
the only contact recorded throughout an open-gripper push was cube-vs-table,
never cube-vs-gripper). No grasp-confirmation logic runs and nothing is ever
locked; this task has nothing to grasp or release. STOPPING already leaves
the gripper open before MANIPULATING starts
(controllers/coordinator.py's STOPPING branch calls
`self.manip.set_gripper(open_=True)`), and APPROACHING/DESCENDING leave it
that way -- closing it earlier than PUSHING (e.g. during the vertical
DESCENDING approach) was observed to produce an uncontrolled glancing knock
in a direction unrelated to the intended push.
"""
from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from .base import Task

if TYPE_CHECKING:
    from controllers.coordinator import TaskCoordinator
    from controllers.manipulation import ManipulationController

# Cartesian target move rate for APPROACHING/DESCENDING/PUSHING (m/s) --
# matches PickAndPlaceTask's _ARM_MOVE_RATE.
_ARM_MOVE_RATE = 0.10  # 10 cm/s

# Hover height above the object during APPROACHING -- same constant
# PickAndPlaceTask uses for its own APPROACHING hover.
_HOVER_Z = 0.15


class PushTask(Task):
    """Walk-to-object, then push it toward a target via real contact
    physics. No grasp/lift/transport/lower/release machinery, and no
    kinematic attachment of any kind -- the object only moves if the EE's
    collision geometry actually shoves it there."""

    class Phase(enum.Enum):
        APPROACHING = "approaching"   # EE hovers above the object
        DESCENDING  = "descending"    # EE drops straight down to push height
        PUSHING     = "pushing"       # EE translates at push height toward target

    def __init__(
        self,
        object_pos: Any,
        target_pos: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        manip: "ManipulationController",
        ftp_offset: float,
        task_cfg: dict[str, Any],
    ) -> None:
        self.model = model
        self.data  = data
        self.manip = manip
        self._ftp  = ftp_offset

        # Object and target positions (world frame)
        self._object_pos = np.array(object_pos, dtype=np.float64)
        self._target_pos = np.array(target_pos, dtype=np.float64)

        # Phase-specific timing/threshold config -- reuses the same keys
        # PickAndPlaceTask/ReachOnlyTask already read; no new config schema.
        self._approach_threshold  = task_cfg.get("approach_threshold",  0.05)
        self._descend_threshold   = task_cfg.get("descend_threshold",   0.025)
        self._min_approach_time   = task_cfg.get("min_approach_time",   1.5)
        self._min_descend_time    = task_cfg.get("min_descend_time",    1.5)
        self._min_transport_time  = task_cfg.get("min_transport_time",  0.8)

        # IK waypoints -- derived from object/target; recomputed after WALKING
        self._wp_approach = None
        self._wp_descend  = None
        self._wp_push     = None
        self._compute_waypoints()

        # Smooth Cartesian interpolation target (updated each step)
        self._arm_interp_target: np.ndarray = np.zeros(3)

        # Object freejoint/body lookup (real-physics read only -- no
        # kinematic attachment, so no qvel address is needed here).
        object_jnt_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._object_qpos_adr = int(model.jnt_qposadr[object_jnt_id])
        self._object_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")

        # Success tolerance. PickAndPlaceTask's _placement_radius (0.12m)
        # was tried first as a baseline and rejected: it is wider than a
        # typical push distance itself, so it would report success on a
        # cube that never moved at all. Measured (reproducibly, see
        # task-5-report.md): a 3cm push closes to within ~2.2cm of target;
        # a 12cm push only closes to within ~11cm (most of the commanded
        # sweep is lost to contact slip). 0.05m is tight enough to demand
        # real movement for a modest push while still being achievable --
        # it is NOT guaranteed to succeed for an arbitrary push distance,
        # which matches the plan's own "Open question" acknowledgment that
        # this may need a real tuning pass rather than a single fixed value.
        self._push_radius = 0.05

        # Internal sub-state machine
        self._phase: PushTask.Phase = PushTask.Phase.APPROACHING
        self._phase_enter_time: float = 0.0

    # ── Public sub-phase accessor (mirrors PickAndPlaceTask.phase) ────────

    @property
    def phase(self) -> "PushTask.Phase":
        return self._phase

    # ── Waypoints ───────────────────────────────────────────────────────────

    def _compute_waypoints(self) -> None:
        ox, oy, oz = self._object_pos
        tx, ty, _tz = self._target_pos
        h = _HOVER_Z
        push_z = oz - self._ftp
        self._wp_approach = np.array([ox, oy, oz + h])
        self._wp_descend  = np.array([ox, oy, push_z])
        self._wp_push      = np.array([tx, ty, push_z])

    def approach_descend_point(self) -> np.ndarray:
        """Return the push/contact-height point used by DESCENDING, for the
        coordinator's generic height-adjustment math in STABILIZING (same
        role as PickAndPlaceTask.approach_descend_point())."""
        return self._wp_descend

    # ── Smooth interp target ────────────────────────────────────────────────

    def _step_interp_target(self, goal: np.ndarray, dt: float) -> np.ndarray:
        """Advance self._arm_interp_target one step toward goal at
        _ARM_MOVE_RATE, mirroring PickAndPlaceTask._step_interp_target."""
        step  = _ARM_MOVE_RATE * dt
        delta = goal - self._arm_interp_target
        dist  = float(np.linalg.norm(delta))
        if dist <= step:
            self._arm_interp_target = goal.copy()
        else:
            self._arm_interp_target += delta / dist * step
        return self._arm_interp_target

    def seed_approach(self, t: float) -> None:
        """Seed the interpolated arm target at the current EE position (no
        jerk on state entry) and seed _phase_enter_time to t.

        Required unconditionally by TaskCoordinator on MANIPULATING entry
        even though it isn't part of the formal Task ABC -- same gap
        ReachOnlyTask (Task 4) and PickAndPlaceTask hit; see
        PickAndPlaceTask.seed_approach for the original rationale."""
        self._arm_interp_target = self.manip.ee_position().copy()
        self._phase_enter_time = t

    # ── Task ABC interface ──────────────────────────────────────────────────

    def _refresh_object_pos(self) -> None:
        pos = self.data.xpos[self._object_body_id].copy()
        self._object_pos[:] = pos
        self._compute_waypoints()

    def target_xy(self) -> np.ndarray:
        self._refresh_object_pos()
        return self._object_pos[:2]

    def _set_phase(self, new_phase: "PushTask.Phase", t: float) -> None:
        print(f"  [t={t:.2f}s] {self._phase.value} -> {new_phase.value}")
        self._phase = new_phase
        self._phase_enter_time = t

    def manip_step(self, coordinator: "TaskCoordinator", t: float, dt: float) -> bool:
        """Advance the APPROACHING -> DESCENDING -> PUSHING sub-state
        machine one step. Returns True once PUSHING's gate passes -- there
        is no RELEASING/LIFTING/LOWERING afterward; the gripper was never
        closed, so there's nothing to release."""
        phase = self._phase

        if phase == PushTask.Phase.APPROACHING:
            current_target = self._step_interp_target(self._wp_approach, dt)
            self.manip.reach_position_smooth(current_target, dt)
            elapsed = t - self._phase_enter_time
            dist    = self.manip.ee_distance_to(self._wp_approach)
            if elapsed >= self._min_approach_time and dist < self._approach_threshold:
                self._set_phase(PushTask.Phase.DESCENDING, t)

        elif phase == PushTask.Phase.DESCENDING:
            current_target = self._step_interp_target(self._wp_descend, dt)
            self.manip.reach_position_smooth(current_target, dt)
            elapsed = t - self._phase_enter_time
            ee_z    = self.manip.ee_position()[2]
            z_err   = abs(ee_z - self._wp_descend[2])
            if elapsed >= self._min_descend_time and z_err < self._descend_threshold:
                self._set_phase(PushTask.Phase.PUSHING, t)

        elif phase == PushTask.Phase.PUSHING:
            # Close the gripper here (PUSHING only -- never during APPROACHING/
            # DESCENDING) to present a flat contact surface. This is not a
            # grasp attempt -- no grasp-confirmation logic runs, nothing is
            # ever locked. It's required for real contact: an open gripper's
            # fingers are spaced wider than the pushable cube and straddle it
            # with zero contact (confirmed via direct contact-log inspection
            # -- the only contact recorded throughout an open-gripper push was
            # cube-vs-table, never cube-vs-gripper). Closing only once PUSHING
            # begins (not during the vertical DESCENDING approach) avoids an
            # uncontrolled glancing knock from the closing motion itself while
            # the EE is still directly above the object -- closing earlier was
            # observed to shove the object sideways in a direction unrelated
            # to the intended push.
            self.manip.set_gripper(open_=False)
            current_target = self._step_interp_target(self._wp_push, dt)
            self.manip.reach_position_smooth(current_target, dt)
            elapsed = t - self._phase_enter_time
            dist    = self.manip.ee_distance_to(self._wp_push)
            if elapsed >= self._min_transport_time and dist < self._approach_threshold:
                return True

        return False

    def is_success(self) -> bool:
        """Real-physics check: the object's actual qpos position (no
        kinematic lock of any kind exists in this task) within _push_radius
        of target_pos's XY."""
        a          = self._object_qpos_adr
        object_pos = self.data.qpos[a:a + 3].copy()
        xy_err     = float(np.linalg.norm(object_pos[:2] - self._target_pos[:2]))
        return xy_err < self._push_radius
