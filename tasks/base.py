"""Abstract base class for pluggable manipulation tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from controllers.coordinator import TaskCoordinator


class Task(ABC):
    """One pluggable manipulation behavior, executed once the robot has
    walked into position. TaskCoordinator owns locomotion/height/return-home;
    a Task owns everything from "arrived" to "this object is done"."""

    @abstractmethod
    def target_xy(self) -> np.ndarray:
        """World XY the coordinator should walk toward (drives WALKING bearing
        + stop_distance check) -- generic locomotion needs this regardless of
        task type."""

    @abstractmethod
    def manip_step(self, coordinator: TaskCoordinator, t: float, dt: float) -> bool:
        """Advance this task's own internal sub-state machine one step.
        Returns True once this task instance is finished (success OR accepted
        failure) and the coordinator should proceed to RETURNING_HOME."""

    @abstractmethod
    def is_success(self) -> bool:
        """Task-level success check (placement_verified-equivalent)."""

    def next_task(self) -> Task | None:
        """Return the next Task to run after this one (for sequencing), or
        None. Default: no sequencing."""
        return None
