"""
High-level task coordinator for the loco-manipulation pick-and-place demo.

State machine:
  INIT -> STANDING -> WALKING -> STOPPING -> STABILIZING
       -> ADJUSTING_HEIGHT                          (adaptive Go2 crouch; skipped if delta=0)
       -> APPROACHING -> DESCENDING -> GRASPING
       -> LIFTING -> TRANSPORTING -> LOWERING -> RELEASING
       -> RETURNING_HOME -> DONE

Key design decisions
--------------------
* Velocity IK (archive m02 pattern): `reach_position_smooth` integrates one
  Jacobian step per physics timestep instead of firing a batch IK every 0.3 s.
  This eliminates the 3 cm position lurches that caused jerky arm motion.

* Full 6-DOF kinematic attachment: at grasp time we record the cube's rotation
  in the EE frame (_grasp_R_local). Each step we reconstruct and impose both the
  world position AND the world quaternion, so the cube cannot tumble during transport.

* Adaptive height (ADJUSTING_HEIGHT): after stabilizing, workspace analysis
  computes a crouch alpha from (a) horizontal arm extension and (b) vertical
  arm extension below base. The locomotion controller blends stand/crouch poses
  without any hardcoded height number --the decision is scenario-driven.

* RETURNING_HOME: after releasing the cube the arm joint-interpolates back to
  the home pose before the simulation ends.
"""
from __future__ import annotations

import enum
import math
from typing import Any

import mujoco
import numpy as np

from .locomotion import GaitMode, LocomotionController
from .manipulation import ManipulationController


class TaskState(enum.Enum):
    INIT             = "init"
    STANDING         = "standing"
    WALKING          = "walking"
    STOPPING         = "stopping"
    STABILIZING      = "stabilizing"       # pause after stopping, arm at home
    ADJUSTING_HEIGHT = "adjusting_height"  # Go2 adaptive crouch for arm workspace
    APPROACHING      = "approaching"       # EE to hover above cube
    DESCENDING       = "descending"        # EE down to cube level
    GRASPING         = "grasping"          # close gripper, hold for contact
    LIFTING          = "lifting"           # raise EE with cube
    TRANSPORTING     = "transporting"      # move EE+cube above placement plate
    LOWERING         = "lowering"          # lower EE+cube to plate level
    RELEASING        = "releasing"         # open gripper, wait
    RETURNING_HOME   = "returning_home"    # arm returns to home pose
    DONE             = "done"


_MIN_STAND_HEIGHT   = 0.22    # m --robot considered stably upright above this
_STOP_VEL_THRESHOLD = 0.06   # m/s --base XY speed to consider "stopped"

# Cartesian target move rate for APPROACHING / TRANSPORTING (m/s).
# The interp target steps at this rate; velocity IK tracks it each timestep.
_ARM_MOVE_RATE = 0.10   # 10 cm/s

# Vertical rate for LIFTING / LOWERING (m/s).
_LIFT_RATE = 0.04       # 4 cm/s → 15 cm lift takes ~3.75 s

# Joint-space return rate for RETURNING_HOME (rad/s in L2 joint space).
# 1.5 rad/s means 1.1 rad total error reaches home in ~0.75 s of commanded travel.
_JOINT_RETURN_RATE = 1.5  # rad/s

# Hover height above cube/plate during approach and transport (m).
_HOVER_Z = 0.15


class TaskCoordinator:
    """Orchestrates locomotion and manipulation for the pick-and-place task."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.data  = data
        self._cfg  = config

        task_cfg = config.get("task", {})

        # Sub-controllers
        self.loco  = LocomotionController(model, data)
        self.manip = ManipulationController(model, data)

        # State machine
        self._state            = TaskState.INIT
        self._stable_since:  float = -999.0
        self._stop_since:    float = -999.0
        self._state_enter_time: float = 0.0

        # Task config
        self._stop_distance       = task_cfg.get("stop_distance",       0.65)
        self._stable_duration     = task_cfg.get("stable_duration",      2.0)
        self._stabilize_duration  = task_cfg.get("stabilize_duration",   2.0)
        self._height_settle_time  = task_cfg.get("height_settle_time",   2.0)
        self._grasp_hold_duration = task_cfg.get("grasp_hold_duration",  1.5)
        self._release_duration    = task_cfg.get("release_duration",     0.8)
        self._approach_threshold  = task_cfg.get("approach_threshold",   0.05)
        self._descend_threshold   = task_cfg.get("descend_threshold",    0.025)
        self._min_approach_time   = task_cfg.get("min_approach_time",    1.5)
        self._min_descend_time    = task_cfg.get("min_descend_time",     1.5)
        self._min_lift_time       = task_cfg.get("min_lift_time",        1.0)
        self._min_transport_time  = task_cfg.get("min_transport_time",   0.8)
        self._min_lower_time      = task_cfg.get("min_lower_time",       1.2)

        # Pickup and placement positions (world frame)
        cube_pos   = task_cfg.get("cube_pos",   [1.6, 0.0,  0.325])
        target_pos = task_cfg.get("target_pos", [1.6, 0.20, 0.331])
        self._cube_pos   = np.array(cube_pos,   dtype=np.float64)
        self._target_pos = np.array(target_pos, dtype=np.float64)

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

        # Commanded joint target for RETURNING_HOME interpolation.
        # Advanced at _JOINT_RETURN_RATE per step independently of measured qpos,
        # so the target races to home quickly regardless of PD tracking lag.
        self._return_q_target: np.ndarray = self.manip.home_qpos()

        # Placement verification tolerances
        self._placement_radius = 0.12
        self._placement_z_tol  = 0.04

    # ── Waypoints ─────────────────────────────────────────────────────────

    def _compute_waypoints(self) -> None:
        cx, cy, cz = self._cube_pos
        tx, ty, tz = self._target_pos
        h = _HOVER_Z
        _FTP = 0.015   # finger pad is ~1.5 cm above ee_site in vertical descent
        self._wp_approach  = np.array([cx, cy, cz + h])
        self._wp_descend   = np.array([cx, cy, cz - _FTP])
        self._wp_lift      = np.array([cx, cy, cz + h])
        self._wp_transport = np.array([tx, ty, tz + h])
        self._wp_lower     = np.array([tx, ty, tz - _FTP])

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

    # ── Adaptive height analysis ───────────────────────────────────────────

    def _compute_height_adjustment(self) -> float:
        """Return crouch alpha (0-1) for the current reach scenario.

        Decision factors:
          - Horizontal reach: arm extends forward to the cube from the robot's
            stopped position. Longer reach -> more forward torque on Go2 body
            -> lower CoM improves balance.
          - Vertical reach below base: arm descends below Go2's base height.
            More downward reach -> lowering base reduces arm extension.

        Both factors are normalised and summed; the result is clamped to [0, 0.5]
        so Go2 never goes beyond a safe ~3 cm lower stance.
        """
        base_z  = float(self.data.qpos[2])
        robot_x = float(self.data.qpos[0])

        horizontal_reach = abs(self._wp_descend[0] - robot_x)
        vertical_reach   = max(0.0, base_z - self._wp_descend[2])

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
            self._refresh_cube_pos()
            if self._base_xy_distance_to_cube() < self._stop_distance:
                self.loco.set_mode(GaitMode.STAND)
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
                # Compute scenario-aware crouch before arm starts moving
                alpha = self._compute_height_adjustment()
                if alpha > 0.02:
                    self.loco.set_crouch_alpha(alpha)
                    print(
                        f"  [t={t:.2f}s] Adaptive height: crouch_alpha={alpha:.2f} "
                        f"(horizontal={abs(self._wp_descend[0]-self.data.qpos[0]):.2f}m "
                        f"below_base={max(0,self.data.qpos[2]-self._wp_descend[2]):.2f}m)"
                    )
                    self._transition(TaskState.ADJUSTING_HEIGHT, t)
                else:
                    # No meaningful height adjustment --proceed directly
                    self._seed_approach(t)
                    self._transition(TaskState.APPROACHING, t)

        elif state == TaskState.ADJUSTING_HEIGHT:
            # Wait for the new stance to settle
            if t - self._state_enter_time >= self._height_settle_time:
                self._seed_approach(t)
                self._transition(TaskState.APPROACHING, t)

        elif state == TaskState.APPROACHING:
            # Velocity IK tracks the incrementally moving interp target each step
            current_target = self._step_interp_target(self._wp_approach, dt)
            self.manip.reach_position_smooth(current_target, dt)
            elapsed = t - self._state_enter_time
            dist    = self.manip.ee_distance_to(self._wp_approach)
            if elapsed >= self._min_approach_time and dist < self._approach_threshold:
                self._arm_interp_target = self.manip.ee_position().copy()
                self._transition(TaskState.DESCENDING, t)

        elif state == TaskState.DESCENDING:
            current_target = self._step_interp_target(self._wp_descend, dt)
            self.manip.reach_position_smooth(current_target, dt)
            elapsed = t - self._state_enter_time
            ee_z    = self.manip.ee_position()[2]
            z_err   = abs(ee_z - self._wp_descend[2])
            if elapsed >= self._min_descend_time and z_err < self._descend_threshold:
                self.manip.set_gripper(open_=False)
                self._transition(TaskState.GRASPING, t)

        elif state == TaskState.GRASPING:
            # Hold firmly at descend position while gripper closes
            self.manip.reach_position_smooth(self._wp_descend, dt)
            self.manip.set_gripper(open_=False)
            if t - self._state_enter_time >= self._grasp_hold_duration:
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
                self._transition(TaskState.LIFTING, t)

        elif state == TaskState.LIFTING:
            if self._grasp_confirmed:
                self._apply_kinematic_attachment()
            # Raise incrementally each step
            self._lift_z_current = min(
                self._lift_z_current + _LIFT_RATE * dt,
                self._wp_lift[2],
            )
            target = np.array([self._wp_lift[0], self._wp_lift[1], self._lift_z_current])
            self.manip.reach_position_smooth(target, dt)
            elapsed   = t - self._state_enter_time
            lift_done = self._lift_z_current >= self._wp_lift[2]
            if elapsed >= self._min_lift_time and lift_done:
                self._arm_interp_target = self.manip.ee_position().copy()
                self._transition(TaskState.TRANSPORTING, t)

        elif state == TaskState.TRANSPORTING:
            if self._grasp_confirmed:
                self._apply_kinematic_attachment()
            current_target = self._step_interp_target(self._wp_transport, dt)
            self.manip.reach_position_smooth(current_target, dt)
            elapsed = t - self._state_enter_time
            dist    = self.manip.ee_distance_to(self._wp_transport)
            if elapsed >= self._min_transport_time and dist < self._approach_threshold:
                self._lift_z_current = float(self.manip.ee_position()[2])
                self._transition(TaskState.LOWERING, t)

        elif state == TaskState.LOWERING:
            if self._grasp_confirmed:
                self._apply_kinematic_attachment()
            self._lift_z_current = max(
                self._lift_z_current - _LIFT_RATE * dt,
                self._wp_lower[2],
            )
            target = np.array([self._wp_lower[0], self._wp_lower[1], self._lift_z_current])
            self.manip.reach_position_smooth(target, dt)
            elapsed    = t - self._state_enter_time
            lower_done = self._lift_z_current <= self._wp_lower[2]
            ee_z       = self.manip.ee_position()[2]
            z_err      = abs(ee_z - self._wp_lower[2])
            if elapsed >= self._min_lower_time and lower_done and z_err < self._descend_threshold:
                self.manip.set_gripper(open_=True)
                self._transition(TaskState.RELEASING, t)

        elif state == TaskState.RELEASING:
            if self._grasp_confirmed:
                self._grasp_confirmed = False
                print(f"  [t={t:.2f}s] Kinematic lock released --cube free")
            self.manip.set_gripper(open_=True)
            if t - self._state_enter_time >= self._release_duration:
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
                self._transition(TaskState.DONE, t)
            else:
                self._return_q_target += delta / dist * step
                self.manip.set_joint_target(self._return_q_target)

        # DONE: hold last position

    # ── Helpers ───────────────────────────────────────────────────────────

    def _seed_approach(self, t: float) -> None:
        """Seed the interpolated arm target at the current EE so the first IK
        step is a zero-length move (no jerk on state entry)."""
        self._arm_interp_target = self.manip.ee_position().copy()

    def _transition(self, new_state: TaskState, t: float) -> None:
        print(f"  [t={t:.2f}s] {self._state.value} -> {new_state.value}")
        self._state            = new_state
        self._state_enter_time = t

    def _refresh_cube_pos(self) -> None:
        pos = self.data.xpos[self._cube_body_id].copy()
        self._cube_pos[:] = pos
        self._compute_waypoints()

    def _base_xy_distance_to_cube(self) -> float:
        return float(np.linalg.norm(self.loco.base_position()[:2] - self._cube_pos[:2]))

    def post_physics_step(self) -> None:
        """Re-enforce 6-DOF kinematic attachment after mj_step.

        The constraint solver can nudge the cube even though we zeroed velocity
        pre-step. Snapping position+orientation post-step keeps the cube firmly
        locked to the gripper in every rendered frame.
        """
        if self._grasp_confirmed:
            self._apply_kinematic_attachment()

    def placement_verified(self) -> bool:
        a        = self._cube_qpos_adr
        cube_pos = self.data.qpos[a:a + 3].copy()
        xy_err   = float(np.linalg.norm(cube_pos[:2] - self._target_pos[:2]))
        z_err    = abs(cube_pos[2] - self._target_pos[2])
        return xy_err < self._placement_radius and z_err < self._placement_z_tol

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def is_done(self) -> bool:
        return self._state == TaskState.DONE

    def status_line(self, t: float) -> str:
        h        = self.loco.base_height()
        dist_base = self._base_xy_distance_to_cube()
        a         = self._cube_qpos_adr
        cube_z    = float(self.data.qpos[a + 2])
        grasped   = self.manip.is_grasped()
        state     = self._state

        if state == TaskState.APPROACHING:
            ee_dist   = self.manip.ee_distance_to(self._wp_approach)
            lbl       = "ee->approach"
        elif state == TaskState.DESCENDING:
            ee_dist   = self.manip.ee_distance_to(self._wp_descend)
            lbl       = "ee->cube"
        elif state in (TaskState.GRASPING, TaskState.LIFTING):
            ee_dist   = self.manip.ee_distance_to(self._wp_lift)
            lbl       = "ee->lift"
        elif state == TaskState.TRANSPORTING:
            ee_dist   = self.manip.ee_distance_to(self._wp_transport)
            lbl       = "ee->transport"
        elif state in (TaskState.LOWERING, TaskState.RELEASING):
            ee_dist   = self.manip.ee_distance_to(self._wp_lower)
            lbl       = "ee->lower"
        elif state == TaskState.RETURNING_HOME:
            ee_dist   = float(np.linalg.norm(
                self.manip.arm_qpos() - self.manip.home_qpos()))
            lbl       = "q_err(home)"
        else:
            ee_dist   = self.manip.ee_distance_to(self._wp_approach)
            lbl       = "ee->approach"

        return (
            f"t={t:5.2f}s | {state.value:18s} | "
            f"h={h:.3f}m | base->cube={dist_base:.3f}m | "
            f"{lbl}={ee_dist:.3f}m | cube_z={cube_z:.3f}m | grasped={grasped}"
        )
