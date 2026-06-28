"""Unit tests for the Task ABC."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.base import Task


class TestTaskAbstractness:
    """Task should not be instantiable directly (it's abstract)."""

    def test_task_cannot_be_instantiated(self):
        """Attempting to instantiate Task directly should raise TypeError."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class Task"):
            Task()

    def test_concrete_subclass_can_be_instantiated(self):
        """A concrete subclass implementing all abstract methods can be instantiated."""

        class MinimalTask(Task):
            def target_xy(self) -> np.ndarray:
                return np.array([1.0, 2.0])

            def manip_step(self, coordinator, t: float, dt: float) -> bool:
                return False

            def is_success(self) -> bool:
                return False

        task = MinimalTask()
        assert task is not None
        assert isinstance(task, Task)

    def test_concrete_subclass_next_task_default_returns_none(self):
        """A concrete subclass should inherit the default next_task() returning None."""

        class MinimalTask(Task):
            def target_xy(self) -> np.ndarray:
                return np.array([1.0, 2.0])

            def manip_step(self, coordinator, t: float, dt: float) -> bool:
                return False

            def is_success(self) -> bool:
                return False

        task = MinimalTask()
        assert task.next_task() is None

    def test_concrete_subclass_can_override_next_task(self):
        """A concrete subclass can override next_task() to return a next task."""

        class MinimalTask(Task):
            def target_xy(self) -> np.ndarray:
                return np.array([1.0, 2.0])

            def manip_step(self, coordinator, t: float, dt: float) -> bool:
                return False

            def is_success(self) -> bool:
                return False

        class ChainedTask(Task):
            def __init__(self, next_task: Task | None = None):
                self._next = next_task

            def target_xy(self) -> np.ndarray:
                return np.array([3.0, 4.0])

            def manip_step(self, coordinator, t: float, dt: float) -> bool:
                return False

            def is_success(self) -> bool:
                return False

            def next_task(self) -> Task | None:
                return self._next

        task1 = MinimalTask()
        task2 = ChainedTask(next_task=task1)

        assert task2.next_task() is task1
        assert task1.next_task() is None

    def test_missing_abstract_methods_raises_typeerror(self):
        """Subclass missing any abstract method should raise TypeError."""

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class IncompleteTask(Task):
                """Missing manip_step and is_success."""

                def target_xy(self) -> np.ndarray:
                    return np.array([1.0, 2.0])

            IncompleteTask()
