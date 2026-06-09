"""Abstract base class for all controllers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import mujoco
import numpy as np


class BaseController(ABC):
    """Minimal interface every controller must implement."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

    @abstractmethod
    def compute(self) -> None:
        """Write control signals directly into self.data.ctrl."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal controller state to defaults."""
        ...

    # ── helpers ──────────────────────────────────────────────────────────

    def _joint_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _actuator_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _site_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    def _get_qpos(self, joint_name: str) -> float:
        jid = self._joint_id(joint_name)
        adr = self.model.jnt_qposadr[jid]
        return float(self.data.qpos[adr])

    def _get_qvel(self, joint_name: str) -> float:
        jid = self._joint_id(joint_name)
        adr = self.model.jnt_dofadr[jid]
        return float(self.data.qvel[adr])

    def _set_ctrl(self, actuator_name: str, value: float) -> None:
        aid = self._actuator_id(actuator_name)
        self.data.ctrl[aid] = value
