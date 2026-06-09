"""
Reach task: Go2 walks to cube, Panda arm reaches toward it.

Wraps TaskCoordinator with logging and success tracking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from controllers.coordinator import TaskCoordinator, TaskState


class ReachTask:
    """Self-contained task runner for the loco-manipulation demo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.data = data
        self.coordinator = TaskCoordinator(model, data, config)
        self._status_interval = config.get("viewer", {}).get("status_interval", 0.5)
        self._last_status_t = -1.0
        self._success_time: float | None = None

    def step(self, dt: float) -> None:
        t = float(self.data.time)
        self.coordinator.step(t, dt)

        # Track first success
        if self._success_time is None and self.coordinator.is_done:
            self._success_time = t
            a = self.coordinator._cube_qpos_adr
            cube_pos = self.data.qpos[a:a + 3].copy()
            placed_ok = self.coordinator.placement_verified()
            print(f"\n  *** PICK-AND-PLACE SUCCESS at t={t:.2f}s ***")
            print(f"  EE position:    {self.coordinator.manip.ee_position()}")
            print(f"  Cube position:  {cube_pos}")
            print(f"  Target plate:   {self.coordinator._target_pos}")
            print(f"  Placement OK:   {placed_ok}\n")

        # Periodic status
        if t - self._last_status_t >= self._status_interval:
            self._last_status_t = t
            print(self.coordinator.status_line(t))

    @property
    def is_done(self) -> bool:
        return self.coordinator.is_done

    @property
    def success_time(self) -> float | None:
        return self._success_time
