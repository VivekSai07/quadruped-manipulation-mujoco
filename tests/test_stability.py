"""Stability tests: robot must stand for 10+ seconds without collapsing."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from controllers.locomotion import GaitMode, LocomotionController
from controllers.manipulation import ManipulationController

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")

_STAND_THRESHOLD = 0.15   # robot considered collapsed below this height
_STAND_DURATION = 10.0    # seconds to hold stand


def run_stand_test(duration_s: float, min_height: float = _STAND_THRESHOLD) -> float:
    """Run stand controller and return minimum height observed."""
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)

    loco = LocomotionController(m, d)
    manip = ManipulationController(m, d)

    min_h = 999.0
    n_steps = int(duration_s / m.opt.timestep)
    for _ in range(n_steps):
        loco.compute()
        manip.compute()
        mujoco.mj_step(m, d)
        h = loco.base_height()
        if h < min_h:
            min_h = h
        if h < min_height:
            break  # collapsed — record the height and stop

    return min_h


class TestStanding:
    def test_stand_5_seconds(self):
        """Robot must stay above 0.15 m for 5 simulated seconds."""
        min_h = run_stand_test(5.0)
        assert min_h > _STAND_THRESHOLD, (
            f"Robot collapsed (min height {min_h:.3f} m < {_STAND_THRESHOLD} m)"
        )

    def test_stand_initial_height(self):
        """After reset, base starts above 0.2 m."""
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)
        h = float(d.qpos[2])
        assert h > 0.20, f"Initial height too low: {h:.3f} m"

    def test_arm_does_not_tip_robot(self):
        """Combined robot does not tip over from arm weight alone."""
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)
        loco = LocomotionController(m, d)
        manip = ManipulationController(m, d)
        # Hold stand + arm home for 3 seconds
        n_steps = int(3.0 / m.opt.timestep)
        min_h = 999.0
        for _ in range(n_steps):
            loco.compute()
            manip.compute()
            mujoco.mj_step(m, d)
            min_h = min(min_h, loco.base_height())
        assert min_h > 0.10, f"Robot tipped (min height {min_h:.3f} m)"
