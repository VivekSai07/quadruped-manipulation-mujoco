"""
Pick-and-place task: Go2 walks to a cube, Panda (or swapped arm) grasps it,
transports it, and places it on a target plate.

This module owns the entire APPROACHING -> RELEASING sub-state machine that
used to live directly on TaskCoordinator. The coordinator only sees this as
"MANIPULATING"; `manip_step()` is called every physics step while in that
state and returns True once the cube has been released and the coordinator
should move on to RETURNING_HOME.

Key design decisions (carried over verbatim from the original coordinator)
----------------------------------------------------------------------------
* Velocity IK (archive m02 pattern): `reach_position_smooth` integrates one
  Jacobian step per physics timestep instead of firing a batch IK every 0.3 s.
  This eliminates the 3 cm position lurches that caused jerky arm motion.

* Full 6-DOF kinematic attachment: at grasp time we record the cube's rotation
  in the EE frame (_grasp_R_local). Each step we reconstruct and impose both the
  world position AND the world quaternion, so the cube cannot tumble during transport.
"""
from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from .base import Task

if TYPE_CHECKING:
    from controllers.coordinator import TaskCoordinator
    from controllers.manipulation import ManipulationController

# Cartesian target move rate for APPROACHING / TRANSPORTING (m/s).
# The interp target steps at this rate; velocity IK tracks it each timestep.
_ARM_MOVE_RATE = 0.10   # 10 cm/s

# Vertical rate for LIFTING / LOWERING (m/s).
_LIFT_RATE = 0.04       # 4 cm/s -> 15 cm lift takes ~3.75 s

# Hover height above cube/plate during approach and transport (m).
_HOVER_Z = 0.15


class PickAndPlaceTask(Task):
    """Walk-to-cube, grasp, transport, and place pick-and-place task."""

    class Phase(enum.Enum):
        APPROACHING  = "approaching"       # EE to hover above cube
        DESCENDING   = "descending"        # EE down to cube level
        GRASPING     = "grasping"          # close gripper, hold for contact
        LIFTING      = "lifting"           # raise EE with cube
        TRANSPORTING = "transporting"      # move EE+cube above placement plate
        LOWERING     = "lowering"          # lower EE+cube to plate level
        RELEASING    = "releasing"         # open gripper, wait

    def __init__(
        self,
        cube_pos: Any,
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

        # Pickup and placement positions (world frame)
        self._cube_pos   = np.array(cube_pos,   dtype=np.float64)
        self._target_pos = np.array(target_pos, dtype=np.float64)

        # Phase-specific timing/threshold config
        self._grasp_hold_duration = task_cfg.get("grasp_hold_duration",  1.5)
        self._release_duration    = task_cfg.get("release_duration",     0.8)
        self._approach_threshold  = task_cfg.get("approach_threshold",   0.05)
        self._descend_threshold   = task_cfg.get("descend_threshold",    0.025)
        self._min_approach_time   = task_cfg.get("min_approach_time",    1.5)
        self._min_descend_time    = task_cfg.get("min_descend_time",     1.5)
        self._min_lift_time       = task_cfg.get("min_lift_time",        1.0)
        self._min_transport_time  = task_cfg.get("min_transport_time",   0.8)
        self._min_lower_time      = task_cfg.get("min_lower_time",       1.2)

        # IK waypoints --derived from cube/target; recomputed after WALKING
        self._wp_approach  = None
        self._wp_descend   = None
        self._wp_lift      = None
        self._wp_transport = None
        self._wp_lower     = None
        self._compute_waypoints()

        # Smooth Cartesian interpolation target (updated each step)
        self._arm_interp_target: np.ndarray = np.zeros(3)

        # Incremental Z for LIFTING / LOWERING
        self._lift_z_current: float = 0.0

        # ── Kinematic attachment (full 6-DOF) ──────────────────────────────
        self._grasp_confirmed: bool = False
        self._grasp_offset:    np.ndarray = np.zeros(3)    # cube_pos - ee_pos
        self._grasp_R_local:   np.ndarray = np.eye(3)      # cube R in EE frame

        # Cube freejoint addresses
        cube_jnt_id           = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_qpos_adr   = int(model.jnt_qposadr[cube_jnt_id])
        self._cube_qvel_adr   = int(model.jnt_dofadr[cube_jnt_id])
        self._cube_body_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")

        # Placement verification tolerances
        self._placement_radius = 0.12
        self._placement_z_tol  = 0.04

        # Internal sub-state machine
        self._phase: PickAndPlaceTask.Phase = PickAndPlaceTask.Phase.APPROACHING
        self._phase_enter_time: float = 0.0

    # ── Public sub-phase accessor (Conflict 1) ─────────────────────────────

    @property
    def phase(self) -> "PickAndPlaceTask.Phase":
        return self._phase

    # ── Waypoints ─────────────────────────────────────────────────────────

    def _compute_waypoints(self) -> None:
        cx, cy, cz = self._cube_pos
        tx, ty, tz = self._target_pos
        h = _HOVER_Z
        self._wp_approach  = np.array([cx, cy, cz + h])
        self._wp_descend   = np.array([cx, cy, cz - self._ftp])
        self._wp_lift      = np.array([cx, cy, cz + h])
        self._wp_transport = np.array([tx, ty, tz + h])
        self._wp_lower     = np.array([tx, ty, tz - self._ftp])

    def approach_descend_point(self) -> np.ndarray:
        """Return the descend waypoint, for the coordinator's generic
        height-adjustment calculation (Conflict 3)."""
        return self._wp_descend

    # ── Smooth interp target ───────────────────────────────────────────────

    def _step_interp_target(self, goal: np.ndarray, dt: float) -> np.ndarray:
        """Advance self._arm_interp_target one step toward goal at _ARM_MOVE_RATE.

        The velocity IK then tracks this moving target each physics timestep,
        giving smooth continuous arm motion instead of periodic 3 cm position jumps.
        """
        step  = _ARM_MOVE_RATE * dt
        delta = goal - self._arm_interp_target
        dist  = float(np.linalg.norm(delta))
        if dist <= step:
            self._arm_interp_target = goal.copy()
        else:
            self._arm_interp_target += delta / dist * step
        return self._arm_interp_target

    def seed_approach(self, t: float) -> None:
        """Seed the interpolated arm target at the current EE so the first IK
        step is a zero-length move (no jerk on state entry), and seed
        _phase_enter_time to the current sim time.

        The coordinator enters MANIPULATING (this task's APPROACHING phase)
        by calling this method rather than via `_set_phase()` -- APPROACHING
        is the __init__-time default phase, never reached through a normal
        phase transition. Without seeding `_phase_enter_time` here it stays
        at its `__init__` default of 0.0, so the first `manip_step` call
        would compute `elapsed = t - 0.0 = t` (absolute sim time) instead of
        true time-in-phase, silently bypassing the `min_approach_time` floor.
        This mirrors what the original coordinator's
        `_transition(TaskState.APPROACHING, t)` did before this port.
        """
        self._arm_interp_target = self.manip.ee_position().copy()
        self._phase_enter_time = t

    # ── 6-DOF kinematic attachment ─────────────────────────────────────────

    @staticmethod
    def _mat2quat(R: np.ndarray) -> np.ndarray:
        """Rotation matrix -> quaternion [w, x, y, z] (Shepperd's method)."""
        t = float(np.trace(R))
        if t > 0.0:
            s = 0.5 / math.sqrt(t + 1.0)
            return np.array([0.25 / s,
                              (R[2,1] - R[1,2]) * s,
                              (R[0,2] - R[2,0]) * s,
                              (R[1,0] - R[0,1]) * s])
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            return np.array([(R[2,1] - R[1,2]) / s, 0.25 * s,
                              (R[0,1] + R[1,0]) / s, (R[0,2] + R[2,0]) / s])
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            return np.array([(R[0,2] - R[2,0]) / s, (R[0,1] + R[1,0]) / s,
                              0.25 * s,               (R[1,2] + R[2,1]) / s])
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            return np.array([(R[1,0] - R[0,1]) / s, (R[0,2] + R[2,0]) / s,
                              (R[1,2] + R[2,1]) / s, 0.25 * s])

    def _apply_kinematic_attachment(self) -> None:
        """Force cube to track EE pose exactly --position AND orientation.

        Records ee_site world position + orientation; reconstructs cube world
        quaternion from the grasp-local rotation captured at GRASPING time.
        Zeroes cube velocity so the physics solver cannot fight the constraint.
        """
        ee_pos  = self.manip.ee_position()
        ee_xmat = self.data.site_xmat[self.manip._ee_site_id].reshape(3, 3)

        new_cube_pos  = ee_pos + self._grasp_offset
        new_cube_R    = ee_xmat @ self._grasp_R_local        # world-frame cube rotation
        new_cube_quat = self._mat2quat(new_cube_R)           # [w, x, y, z]

        a = self._cube_qpos_adr
        self.data.qpos[a:a + 3] = new_cube_pos
        self.data.qpos[a + 3:a + 7] = new_cube_quat
        v = self._cube_qvel_adr
        self.data.qvel[v:v + 6] = 0.0

    def post_physics_step(self) -> None:
        """Re-enforce 6-DOF kinematic attachment after mj_step.

        The constraint solver can nudge the cube even though we zeroed velocity
        pre-step. Snapping position+orientation post-step keeps the cube firmly
        locked to the gripper in every rendered frame.
        """
        if self._grasp_confirmed:
            self._apply_kinematic_attachment()

    # ── Task ABC interface ───────────────────────────────────────────────

    def _refresh_cube_pos(self) -> None:
        pos = self.data.xpos[self._cube_body_id].copy()
        self._cube_pos[:] = pos
        self._compute_waypoints()

    def target_xy(self) -> np.ndarray:
        self._refresh_cube_pos()
        return self._cube_pos[:2]

    def _set_phase(self, new_phase: "PickAndPlaceTask.Phase", t: float) -> None:
        print(f"  [t={t:.2f}s] {self._phase.value} -> {new_phase.value}")
        self._phase = new_phase
        self._phase_enter_time = t

    def manip_step(self, coordinator: "TaskCoordinator", t: float, dt: float) -> bool:
        """Advance the APPROACHING..RELEASING sub-state machine one step.

        Returns True once RELEASING's duration check passes (today's
        transition into RETURNING_HOME).
        """
        phase = self._phase

        if phase == PickAndPlaceTask.Phase.APPROACHING:
            # Velocity IK tracks the incrementally moving interp target each step
            current_target = self._step_interp_target(self._wp_approach, dt)
            self.manip.reach_position_smooth(current_target, dt)
            elapsed = t - self._phase_enter_time
            dist    = self.manip.ee_distance_to(self._wp_approach)
            if elapsed >= self._min_approach_time and dist < self._approach_threshold:
                self._arm_interp_target = self.manip.ee_position().copy()
                self._set_phase(PickAndPlaceTask.Phase.DESCENDING, t)

        elif phase == PickAndPlaceTask.Phase.DESCENDING:
            current_target = self._step_interp_target(self._wp_descend, dt)
            self.manip.reach_position_smooth(current_target, dt)
            elapsed = t - self._phase_enter_time
            ee_z    = self.manip.ee_position()[2]
            z_err   = abs(ee_z - self._wp_descend[2])
            if elapsed >= self._min_descend_time and z_err < self._descend_threshold:
                self.manip.set_gripper(open_=False)
                self._set_phase(PickAndPlaceTask.Phase.GRASPING, t)

        elif phase == PickAndPlaceTask.Phase.GRASPING:
            # Hold firmly at descend position while gripper closes
            self.manip.reach_position_smooth(self._wp_descend, dt)
            self.manip.set_gripper(open_=False)
            if t - self._phase_enter_time >= self._grasp_hold_duration:
                if self.manip.is_grasped():
                    ee_pos   = self.manip.ee_position()
                    ee_xmat  = self.data.site_xmat[self.manip._ee_site_id].reshape(3, 3)
                    a        = self._cube_qpos_adr
                    cube_pos = self.data.qpos[a:a + 3].copy()
                    cube_xmat = self.data.xmat[self._cube_body_id].reshape(3, 3)

                    self._grasp_offset  = cube_pos - ee_pos
                    self._grasp_R_local = ee_xmat.T @ cube_xmat  # cube rot in EE frame
                    self._grasp_confirmed = True
                    print(
                        f"  [t={t:.2f}s] Grasp confirmed --6DOF lock engaged "
                        f"(offset={self._grasp_offset.round(4)})"
                    )
                else:
                    self._grasp_confirmed = False
                    print(f"  [t={t:.2f}s] WARNING: no contact --lifting without lock")

                self._lift_z_current = float(self.manip.ee_position()[2])
                self._set_phase(PickAndPlaceTask.Phase.LIFTING, t)

        elif phase == PickAndPlaceTask.Phase.LIFTING:
            if self._grasp_confirmed:
                self._apply_kinematic_attachment()
            # Raise incrementally each step
            self._lift_z_current = min(
                self._lift_z_current + _LIFT_RATE * dt,
                self._wp_lift[2],
            )
            target = np.array([self._wp_lift[0], self._wp_lift[1], self._lift_z_current])
            self.manip.reach_position_smooth(target, dt)
            elapsed   = t - self._phase_enter_time
            lift_done = self._lift_z_current >= self._wp_lift[2]
            if elapsed >= self._min_lift_time and lift_done:
                self._arm_interp_target = self.manip.ee_position().copy()
                self._set_phase(PickAndPlaceTask.Phase.TRANSPORTING, t)

        elif phase == PickAndPlaceTask.Phase.TRANSPORTING:
            if self._grasp_confirmed:
                self._apply_kinematic_attachment()
            current_target = self._step_interp_target(self._wp_transport, dt)
            self.manip.reach_position_smooth(current_target, dt)
            elapsed = t - self._phase_enter_time
            dist    = self.manip.ee_distance_to(self._wp_transport)
            if elapsed >= self._min_transport_time and dist < self._approach_threshold:
                self._lift_z_current = float(self.manip.ee_position()[2])
                self._set_phase(PickAndPlaceTask.Phase.LOWERING, t)

        elif phase == PickAndPlaceTask.Phase.LOWERING:
            if self._grasp_confirmed:
                self._apply_kinematic_attachment()
            self._lift_z_current = max(
                self._lift_z_current - _LIFT_RATE * dt,
                self._wp_lower[2],
            )
            target = np.array([self._wp_lower[0], self._wp_lower[1], self._lift_z_current])
            self.manip.reach_position_smooth(target, dt)
            elapsed    = t - self._phase_enter_time
            lower_done = self._lift_z_current <= self._wp_lower[2]
            ee_z       = self.manip.ee_position()[2]
            z_err      = abs(ee_z - self._wp_lower[2])
            if elapsed >= self._min_lower_time and lower_done and z_err < self._descend_threshold:
                self.manip.set_gripper(open_=True)
                self._set_phase(PickAndPlaceTask.Phase.RELEASING, t)

        elif phase == PickAndPlaceTask.Phase.RELEASING:
            if self._grasp_confirmed:
                self._grasp_confirmed = False
                print(f"  [t={t:.2f}s] Kinematic lock released --cube free")
            self.manip.set_gripper(open_=True)
            if t - self._phase_enter_time >= self._release_duration:
                return True

        return False

    def is_success(self) -> bool:
        a        = self._cube_qpos_adr
        cube_pos = self.data.qpos[a:a + 3].copy()
        xy_err   = float(np.linalg.norm(cube_pos[:2] - self._target_pos[:2]))
        z_err    = abs(cube_pos[2] - self._target_pos[2])
        return xy_err < self._placement_radius and z_err < self._placement_z_tol
