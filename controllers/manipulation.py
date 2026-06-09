"""
Manipulation controller for the Franka Panda arm.

The Panda actuators use integrated PD control (MuJoCo `general` type with
gainprm/biasprm). The ctrl input is the desired joint angle (rad); MuJoCo
internally computes: force = kp*(ctrl - q) + kd*(0 - q_dot).

So this controller only needs to:
  1. Track a desired joint configuration in ctrl[12:20]
  2. Optionally compute joint targets via numerical IK

Joint layout in combined.xml actuators (indices 12-19):
  ctrl[12] → actuator1 → joint1   (z-axis, base rotation)
  ctrl[13] → actuator2 → joint2   (shoulder pitch)
  ctrl[14] → actuator3 → joint3   (upper arm roll)
  ctrl[15] → actuator4 → joint4   (elbow pitch)
  ctrl[16] → actuator5 → joint5   (forearm roll)
  ctrl[17] → actuator6 → joint6   (wrist pitch)
  ctrl[18] → actuator7 → joint7   (wrist roll)
  ctrl[19] → actuator8 → gripper  (0=open, 255=closed)
"""
from __future__ import annotations

import math
from typing import Sequence

import mujoco
import numpy as np

from .base import BaseController

# Panda home pose: arm folded safely above robot
_HOME_POSE = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853], dtype=np.float64)

# Velocity-IK gains — archive m02 style, called every physics step
_VEL_IK_KP:  float = 5.0   # translational gain (m/s per m of error)
_VEL_IK_KR:  float = 3.0   # rotational gain   (rad/s per rad of error)
_VEL_IK_MAX: float = 0.8   # max joint velocity (rad/s) — prevents violent arm lurches

# Target EE orientation for ALL grasp waypoints: gripper pointing straight DOWN.
# Columns are the local-X, local-Y, local-Z axes expressed in world coordinates.
#   local-Z (approach direction) = world [0, 0, -1]  → points straight down ✓
#   local-X                      = world [1,  0,  0]
#   local-Y (finger-spread axis) = world [0, -1,  0]  → fingers squeeze in ±Y world
# This matches the archive's "fixed_rotation" pattern (m02/m13 mark series).
_TARGET_ROTATION = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)

# Rotation error gain relative to position error in the 6D IK.
# Keep lower than 1.0 so position convergence is not sacrificed for orientation.
_IK_ROT_GAIN = 0.4

# Joint limits (from panda.xml)
_Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_Q_HI = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

_ARM_ACTUATOR_NAMES = [
    "actuator1", "actuator2", "actuator3", "actuator4",
    "actuator5", "actuator6", "actuator7",
]
_GRIPPER_ACTUATOR = "actuator8"

_ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
]


class ManipulationController(BaseController):
    """Joint-space position controller with optional numerical IK for the Panda arm."""

    IK_ITERATIONS: int = 50
    IK_STEP_SIZE: float = 0.5
    IK_DAMPING: float = 0.01      # Levenberg-Marquardt damping
    IK_TOLERANCE: float = 0.003   # 3 mm convergence threshold

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        self._q_target = _HOME_POSE.copy()
        self._gripper_open = True

        # Cached IDs
        self._act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _ARM_ACTUATOR_NAMES
        ]
        self._gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR
        )
        self._jnt_qadr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in _ARM_JOINT_NAMES
        ]
        self._jnt_dadr = [
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in _ARM_JOINT_NAMES
        ]
        self._ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )

        # Body IDs for contact-based grasp detection (from archive GraspController)
        self._left_finger_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "panda_left_finger"
        )
        self._right_finger_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "panda_right_finger"
        )
        self._cube_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "target_cube"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def set_joint_target(self, q: np.ndarray) -> None:
        """Set desired arm joint configuration directly (7 values)."""
        self._q_target = np.clip(q, _Q_LO, _Q_HI)

    def set_home(self) -> None:
        self._q_target = _HOME_POSE.copy()

    def set_gripper(self, open_: bool) -> None:
        self._gripper_open = open_

    def reach_position(self, target_world: np.ndarray) -> bool:
        """Batch position IK — converges in one call. Use for instantaneous jumps."""
        q_ik, converged = self._numerical_ik(target_world)
        self._q_target = q_ik
        return converged

    def reach_position_smooth(
        self,
        target_world: np.ndarray,
        dt: float,
        target_rot: np.ndarray | None = _TARGET_ROTATION,
    ) -> bool:
        """Velocity IK — one integration step per call (archive m02 pattern).

        Drive arm joints smoothly toward target_world at _VEL_IK_KP m/s per m
        of error. Call every physics timestep for continuous, jerk-free motion.
        Returns True once EE is within IK_TOLERANCE of target.
        """
        col_ids = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]
            for n in _ARM_JOINT_NAMES
        ]

        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        ctrl_save = self.data.ctrl.copy()

        # FK at measured arm position for accurate Jacobian
        q_meas = self.arm_qpos()
        for i, adr in enumerate(self._jnt_qadr):
            self.data.qpos[adr] = q_meas[i]
        mujoco.mj_fwdPosition(self.model, self.data)

        ee_pos   = self.data.site_xpos[self._ee_site_id].copy()
        pos_err  = target_world - ee_pos
        pos_norm = float(np.linalg.norm(pos_err))

        converged = pos_norm < self.IK_TOLERANCE
        if not converged:
            use_rot = target_rot is not None
            n_task  = 6 if use_rot else 3
            damp_e  = self.IK_DAMPING ** 2 * np.eye(n_task)

            J_full = np.zeros((6, self.model.nv))
            mujoco.mj_jacSite(
                self.model, self.data, J_full[:3], J_full[3:], self._ee_site_id
            )
            J_arm = J_full[:n_task, col_ids]   # (n_task × 7)

            if use_rot:
                R_curr  = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
                rot_err = self._rotation_error(R_curr, target_rot)
                v_des   = np.concatenate([
                    _VEL_IK_KP * pos_err,
                    _VEL_IK_KR * _IK_ROT_GAIN * rot_err,
                ])
            else:
                v_des = _VEL_IK_KP * pos_err

            dq_vel = J_arm.T @ np.linalg.solve(J_arm @ J_arm.T + damp_e, v_des)
            dq_vel = np.clip(dq_vel, -_VEL_IK_MAX, _VEL_IK_MAX)

            # Integrate from the COMMANDED target — avoids PD-lag wind-up
            q_new = np.clip(self._q_target + dq_vel * dt, _Q_LO, _Q_HI)
            self._q_target = q_new

        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.ctrl[:] = ctrl_save
        mujoco.mj_fwdPosition(self.model, self.data)
        return converged

    def home_qpos(self) -> np.ndarray:
        """Return the arm's home joint configuration."""
        return _HOME_POSE.copy()

    def reset(self) -> None:
        self._q_target = _HOME_POSE.copy()
        self._gripper_open = True

    def compute(self) -> None:
        # Arm: set ctrl to desired position (Panda actuators integrate PD internally)
        for i, aid in enumerate(self._act_ids):
            self.data.ctrl[aid] = float(self._q_target[i])
        # Gripper: ctrl=255 → open (0.04 m gap), ctrl=0 → closed (spring pulls to 0)
        self.data.ctrl[self._gripper_id] = 255.0 if self._gripper_open else 0.0

    # ── State queries ─────────────────────────────────────────────────────

    def ee_position(self) -> np.ndarray:
        """End-effector site position in world frame."""
        mujoco.mj_fwdPosition(self.model, self.data)
        return self.data.site_xpos[self._ee_site_id].copy()

    def ee_distance_to(self, target: np.ndarray) -> float:
        return float(np.linalg.norm(self.ee_position() - target))

    def arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self._jnt_qadr])

    def arm_qvel(self) -> np.ndarray:
        return np.array([self.data.qvel[adr] for adr in self._jnt_dadr])

    def is_grasped(self) -> bool:
        """Return True if both fingers are in contact with the cube (contact physics)."""
        left_touch = False
        right_touch = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            pair = {b1, b2}
            if pair == {self._left_finger_id, self._cube_body_id}:
                left_touch = True
            if pair == {self._right_finger_id, self._cube_body_id}:
                right_touch = True
        return left_touch and right_touch

    # ── Numerical IK (6-DOF Jacobian pseudo-inverse) ─────────────────────

    @staticmethod
    def _rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
        """Return the 3D rotation-error vector (world frame) from R_current to R_target.

        Uses the axis-angle representation of R_target @ R_current^T.
        The result has the same sign convention as MuJoCo's angular-velocity Jacobian.
        """
        R_err = R_target @ R_current.T
        cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        theta = float(np.arccos(cos_theta))
        if abs(theta) < 1e-7:
            return np.zeros(3)
        factor = theta / (2.0 * math.sin(theta))
        return factor * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])

    def _numerical_ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = _TARGET_ROTATION,
    ) -> tuple[np.ndarray, bool]:
        """6-DOF Levenberg-Marquardt IK (position + optional orientation).

        Keeping orientation fixed across all arm waypoints (approach, descend,
        grasp, lift, transport, lower) replicates the archive mark-series pattern
        of `fixed_rotation = home_pose.rotation` — the EE always approaches the
        cube perpendicularly from above.
        """
        q = self.arm_qpos().copy()

        # Pre-compute arm DOF column indices once
        col_ids = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]
            for n in _ARM_JOINT_NAMES
        ]

        # Snapshot physics state — IK only uses forward kinematics
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        ctrl_save = self.data.ctrl.copy()

        use_rot = target_rot is not None
        n_task = 6 if use_rot else 3
        damp_eye = self.IK_DAMPING ** 2 * np.eye(n_task)

        converged = False
        for _ in range(self.IK_ITERATIONS):
            for i, adr in enumerate(self._jnt_qadr):
                self.data.qpos[adr] = q[i]
            mujoco.mj_fwdPosition(self.model, self.data)

            ee_pos = self.data.site_xpos[self._ee_site_id].copy()
            pos_err = target_pos - ee_pos
            err_norm = float(np.linalg.norm(pos_err))

            if err_norm < self.IK_TOLERANCE:
                converged = True
                break

            # Full 6×nv Jacobian (position rows + angular-velocity rows)
            J_full = np.zeros((6, self.model.nv))
            mujoco.mj_jacSite(
                self.model, self.data, J_full[:3], J_full[3:], self._ee_site_id
            )
            J_arm = J_full[:n_task, col_ids]  # (n_task × 7)

            if use_rot:
                R_curr = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
                rot_err = self._rotation_error(R_curr, target_rot)
                err = np.concatenate([pos_err, _IK_ROT_GAIN * rot_err])
            else:
                err = pos_err

            # Levenberg-Marquardt: dq = J^T (J J^T + λ²I)^{-1} err
            JJT = J_arm @ J_arm.T
            dq = J_arm.T @ np.linalg.solve(JJT + damp_eye, err)

            q = q + self.IK_STEP_SIZE * dq
            q = np.clip(q, _Q_LO, _Q_HI)

        # Restore physics state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.ctrl[:] = ctrl_save
        mujoco.mj_fwdPosition(self.model, self.data)

        return q, converged
