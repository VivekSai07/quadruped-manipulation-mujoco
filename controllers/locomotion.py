"""
Locomotion controller for the Unitree Go2.

Two modes:
  STAND  — PD tracking of a fixed standing pose
  TROT   — sinusoidal trot gait with Bézier foot trajectories

Actuator mapping (matches combined.xml actuator order):
  ctrl[0]  FR_hip    ctrl[1]  FR_thigh   ctrl[2]  FR_calf
  ctrl[3]  FL_hip    ctrl[4]  FL_thigh   ctrl[5]  FL_calf
  ctrl[6]  RR_hip    ctrl[7]  RR_thigh   ctrl[8]  RR_calf
  ctrl[9]  RL_hip    ctrl[10] RL_thigh   ctrl[11] RL_calf

PD formula applied to each motor:
  τ = kp * (q_des - q) + kd * (0 - q_dot)
"""
from __future__ import annotations

import enum
import math
from typing import Any

import mujoco
import numpy as np

from .base import BaseController


class GaitMode(enum.Enum):
    STAND = "stand"
    TROT = "trot"


# ── Leg joint names (FR, FL, RR, RL) in actuator order ────────────────────
_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

_ACTUATOR_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

# Default standing pose: [hip_ab, hip, knee] × 4 legs (FR, FL, RR, RL)
# From unitree_mujoco stand_go2.py example
_STAND_POSE = np.array([
     0.006,  0.609, -1.218,   # FR
    -0.006,  0.609, -1.218,   # FL
     0.006,  0.609, -1.218,   # RR
    -0.006,  0.609, -1.218,   # RL
], dtype=np.float64)

# Crouched pose — thigh more pitched, knee more bent → body ~6 cm lower.
# Used for adaptive height during manipulation; blended via set_crouch_alpha().
_CROUCH_POSE = np.array([
     0.006,  0.80, -1.50,    # FR
    -0.006,  0.80, -1.50,    # FL
     0.006,  0.80, -1.50,    # RR
    -0.006,  0.80, -1.50,    # RL
], dtype=np.float64)

# PD gains tuned for Go2 + Panda combined weight (~21 kg)
_KP_HIP   = 60.0   # abduction (hip_ab)
_KP_THIGH = 80.0   # hip fore/aft
_KP_KNEE  = 100.0  # knee
_KD       = 4.0    # unified derivative gain

_KP = np.array([
    _KP_HIP, _KP_THIGH, _KP_KNEE,
    _KP_HIP, _KP_THIGH, _KP_KNEE,
    _KP_HIP, _KP_THIGH, _KP_KNEE,
    _KP_HIP, _KP_THIGH, _KP_KNEE,
], dtype=np.float64)

_KD_VEC = np.full(12, _KD, dtype=np.float64)


class LocomotionController(BaseController):
    """PD locomotion controller for the Go2 quadruped."""

    # Trot gait parameters
    GAIT_PERIOD: float = 0.6       # seconds per full gait cycle
    SWING_HEIGHT: float = 0.05     # max foot lift height (m)
    STRIDE_X: float = 0.10         # half stride length (m) — forward displacement
    _WALK_RAMP: float = 3.0        # seconds to ramp to full stride at walk start

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> None:
        super().__init__(model, data)
        self._kp = kp if kp is not None else _KP.copy()
        self._kd = kd if kd is not None else _KD_VEC.copy()
        self._mode = GaitMode.STAND
        self._gait_t0: float = 0.0    # sim time when TROT started
        self._stand_t0: float = 0.0
        self._stand_ramp: float = 2.0  # seconds to ramp to stand from lie-down
        self._crouch_alpha: float = 0.0  # 0 = full stand, 1 = full crouch

        # Cached actuator + joint DOF addresses (resolved once at init)
        self._act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in _ACTUATOR_NAMES
        ]
        self._jnt_qadr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in _JOINT_NAMES
        ]
        self._jnt_dadr = [
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in _JOINT_NAMES
        ]

    # ── Public API ────────────────────────────────────────────────────────

    def set_mode(self, mode: GaitMode) -> None:
        if mode == GaitMode.TROT and self._mode != GaitMode.TROT:
            self._gait_t0 = self.data.time
        self._mode = mode

    def set_crouch_alpha(self, alpha: float) -> None:
        """Blend standing height toward crouch (0 = stand, 1 = ~6 cm lower).

        The target pose is: (1-alpha)*_STAND_POSE + alpha*_CROUCH_POSE.
        Clamped to [0, 1]. Takes effect immediately via the PD stand controller.
        """
        self._crouch_alpha = float(np.clip(alpha, 0.0, 1.0))

    def reset(self) -> None:
        self._mode = GaitMode.STAND
        self._gait_t0 = 0.0

    def compute(self) -> None:
        if self._mode == GaitMode.STAND:
            self._compute_stand()
        else:
            self._compute_trot()

    # ── Standing controller ───────────────────────────────────────────────

    def _compute_stand(self) -> None:
        q = self._read_q()
        qd = self._read_qd()
        q_des = self._stand_target()
        tau = self._kp * (q_des - q) + self._kd * (0.0 - qd)
        self._write_ctrl(tau)

    def _stand_target(self) -> np.ndarray:
        """Smooth ramp from keyframe home to stand pose, with optional crouch blend."""
        t_elapsed = self.data.time - self._stand_t0
        ramp = min(1.0, t_elapsed / self._stand_ramp)
        # Keyframe home: [0, 0.9, -1.8] per leg (partially folded)
        home = np.array([0, 0.9, -1.8] * 4, dtype=np.float64)
        # Blend stand with crouch according to _crouch_alpha
        target_stand = (1.0 - self._crouch_alpha) * _STAND_POSE + self._crouch_alpha * _CROUCH_POSE
        return (1.0 - ramp) * home + ramp * target_stand

    # ── Trot gait controller ──────────────────────────────────────────────

    def _compute_trot(self) -> None:
        t_gait = self.data.time - self._gait_t0
        # Ramp amplitude for smooth gait onset
        ramp = min(1.0, t_gait / self._WALK_RAMP)

        q = self._read_q()
        qd = self._read_qd()
        q_des = self._trot_joint_targets(t_gait, ramp)
        tau = self._kp * (q_des - q) + self._kd * (0.0 - qd)
        self._write_ctrl(tau)

    def _trot_joint_targets(self, t: float, ramp: float) -> np.ndarray:
        """
        Sinusoidal trot gait in joint space.

        Diagonal pairs:
          Phase 0 (FR, RL): swing at sin > 0
          Phase 1 (FL, RR): swing at sin < 0

        We modulate the thigh and knee joints sinusoidally to lift and
        advance each foot pair in alternation.
        """
        phi = (2.0 * math.pi * t) / self.GAIT_PERIOD

        # Phase offset: FR+RL swing together, FL+RR swing together
        # FR=0, FL=π, RR=π, RL=0  (diagonal pairs)
        phases = [0.0, math.pi, math.pi, 0.0]  # FR, FL, RR, RL

        q_des = _STAND_POSE.copy()

        for i, phase_off in enumerate(phases):
            phi_leg = phi + phase_off
            sin_val = math.sin(phi_leg)

            # +thigh angle = foot moves backward; negate so swing (sin>0) reaches forward
            q_des[3 * i + 1] = _STAND_POSE[3 * i + 1] - ramp * 0.15 * sin_val

            # Flex knee during swing (subtract → more negative = foot lifts), extend during stance
            q_des[3 * i + 2] = _STAND_POSE[3 * i + 2] - ramp * 0.20 * sin_val

            # Hip abduction: lateral sway for balance
            q_des[3 * i + 0] = _STAND_POSE[3 * i + 0] + ramp * 0.04 * math.cos(phi_leg)

        return q_des

    # ── Helpers ───────────────────────────────────────────────────────────

    def _read_q(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self._jnt_qadr])

    def _read_qd(self) -> np.ndarray:
        return np.array([self.data.qvel[adr] for adr in self._jnt_dadr])

    def _write_ctrl(self, tau: np.ndarray) -> None:
        for i, aid in enumerate(self._act_ids):
            self.data.ctrl[aid] = float(tau[i])

    def base_height(self) -> float:
        """Return world-frame z-height of the Go2 base_link via freejoint qpos[2].
        qpos[2] is always valid after mj_resetDataKeyframe; no forward pass needed."""
        return float(self.data.qpos[2])

    def base_velocity(self) -> np.ndarray:
        """Base linear velocity in world frame (3-vector)."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        # freejoint dof offset: body 1 after world, dof 0-2 linear
        # qvel[0:3] are linear velocity of the freejoint body
        return self.data.qvel[0:3].copy()

    def base_position(self) -> np.ndarray:
        """Base position in world frame (3-vector)."""
        return self.data.qpos[0:3].copy()
