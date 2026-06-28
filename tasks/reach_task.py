"""
Reach task: Go2 walks to cube, Panda arm reaches toward it.

Wraps TaskCoordinator with logging and success tracking.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

from controllers.arms import DEFAULT_ARM
from controllers.coordinator import TaskCoordinator, TaskState
from controllers.end_effectors import DEFAULT_END_EFFECTOR

if TYPE_CHECKING:
    from controllers.manipulation import ManipulationController
    from tasks.base import Task

# Signature for the construction-order-safe task factory: called only after
# TaskCoordinator (and therefore the real, shared ManipulationController) has
# been constructed, so the Task it returns is guaranteed to operate on the
# exact same manip instance TaskCoordinator.step() drives every tick. See
# .superpowers/sdd/task-7-brief.md for why a pre-built Task instance can't be
# accepted directly here -- it would risk a second, independent
# ManipulationController whose commands never reach the simulation.
TaskFactory = Callable[["ManipulationController", float, dict[str, Any]], "Task"]


class ReachTask:
    """Self-contained task runner for the loco-manipulation demo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: dict[str, Any],
        arm: str = DEFAULT_ARM,
        end_effector: str = DEFAULT_END_EFFECTOR,
        task_factory: TaskFactory | None = None,
    ) -> None:
        self.model = model
        self.data = data
        # Build the coordinator first -- this is what gives us the real,
        # shared ManipulationController/ftp_offset. Only after this exists
        # can a caller-supplied task_factory be resolved into a Task that is
        # guaranteed to share that exact manip instance (see TaskFactory
        # docstring above / task-7-brief.md's construction-order gotcha).
        self.coordinator = TaskCoordinator(model, data, config, arm=arm, end_effector=end_effector)
        if task_factory is not None:
            task_cfg = config.get("task", {})
            custom_task = task_factory(self.coordinator.manip, self.coordinator._ftp, task_cfg)
            self.coordinator._active_task = custom_task
        self._status_interval = config.get("viewer", {}).get("status_interval", 0.5)
        self._last_status_t = -1.0
        self._success_time: float | None = None

    def step(self, dt: float) -> None:
        t = float(self.data.time)
        self.coordinator.step(t, dt)

        # Track first success
        if self._success_time is None and self.coordinator.is_done:
            self._success_time = t
            active = self.coordinator.active_task
            task_name = type(active).__name__
            placed_ok = active.is_success()
            print(f"\n  *** {task_name} SUCCESS at t={t:.2f}s ***")
            print(f"  EE position:    {self.coordinator.manip.ee_position()}")
            # Task-specific position detail is only available on some task
            # types (PickAndPlaceTask/PushTask have _target_pos; ReachOnlyTask
            # has _target_point; chained sequences may differ again) -- guard
            # with getattr so this never crashes regardless of which of the
            # four Task shapes (or chain thereof) is active.
            cube_adr = getattr(active, "_cube_qpos_adr", None)
            if cube_adr is not None:
                cube_pos = self.data.qpos[cube_adr:cube_adr + 3].copy()
                print(f"  Object position: {cube_pos}")
            target = getattr(active, "_target_pos", None)
            if target is None:
                target = getattr(active, "_target_point", None)
            if target is not None:
                print(f"  Target:         {target}")
            print(f"  Success:        {placed_ok}\n")

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
