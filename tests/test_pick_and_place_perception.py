"""Vision Task 4: integration-seam tests for PickAndPlaceTask's optional
`cube_detector` constructor parameter.

These tests use tiny stub test doubles (NOT a real perception.cube_detector.
CubeDetector) so they stay fast and isolated -- no camera rendering involved.
They confirm the seam added to `_refresh_cube_pos()`:
  - a configured stub detector's position is what `target_xy()` reports
    pre-grasp (proving the seam actually routes through the detector);
  - a detector that returns None falls back to ground-truth `data.xpos`
    (transient-miss fallback path);
  - the default (`cube_detector=None`, the parameter omitted entirely)
    behaves identically to pre-Task-4 behavior -- ground truth used directly.

`target_xy()` itself, `is_success()`, and all grasp-confirmation/kinematic-
attachment code are unchanged by this phase; these tests only exercise the
new, defaulted `cube_detector` seam.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.pick_and_place import PickAndPlaceTask

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")
CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))


class _StubDetector:
    """Always returns a fixed, deliberately-wrong-vs-ground-truth position."""

    def __init__(self, fixed_pos):
        self._fixed_pos = np.array(fixed_pos, dtype=np.float64)
        self.calls = 0

    def detect(self, data):
        self.calls += 1
        return self._fixed_pos.copy()


class _NoneDetector:
    """Simulates a transient detection miss -- always returns None."""

    def detect(self, data):
        return None


class TestPickAndPlaceCubeDetectorSeam:
    """Direct-construction tests (mirroring TestRegraspFaultRecovery's
    pattern in tests/test_task.py) -- no full TaskCoordinator pipeline
    needed to exercise this seam."""

    def _build_task(self, cfg, cube_detector=None, _omit_param=False):
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)

        from controllers.manipulation import ManipulationController  # noqa: PLC0415

        manip = ManipulationController(m, d)
        ee_home = manip.ee_position().copy()

        hover_z = 0.15
        ftp_offset = manip._ee_spec.ftp_offset
        cube_pos = [float(ee_home[0]), float(ee_home[1]), float(ee_home[2] - hover_z)]
        target_pos = [float(ee_home[0]) + 0.1, float(ee_home[1]), float(ee_home[2] - hover_z) + 0.006]

        cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qadr = int(m.jnt_qposadr[cube_jid])
        d.qpos[cube_qadr:cube_qadr + 3] = cube_pos
        mujoco.mj_forward(m, d)

        task_cfg = dict(cfg["task"])
        if _omit_param:
            task = PickAndPlaceTask(cube_pos, target_pos, m, d, manip, ftp_offset, task_cfg)
        else:
            task = PickAndPlaceTask(
                cube_pos, target_pos, m, d, manip, ftp_offset, task_cfg,
                cube_detector=cube_detector,
            )
        return task, m, d, manip, cube_pos

    def test_target_xy_uses_stub_detector_position_pre_grasp(self, cfg):
        """Pre-grasp, target_xy() must report the STUB's fixed XY -- not
        ground truth -- proving the seam routes through the detector."""
        stub_pos = [0.42, -0.17, 0.30]   # deliberately far from real cube_pos
        stub = _StubDetector(stub_pos)
        task, m, d, manip, real_cube_pos = self._build_task(cfg, cube_detector=stub)

        assert not task._grasp_confirmed

        xy = task.target_xy()

        assert np.allclose(xy, np.array(stub_pos[:2]))
        assert not np.allclose(xy, np.array(real_cube_pos[:2]))
        assert stub.calls == 1

    def test_target_xy_falls_back_to_ground_truth_on_none_detection(self, cfg):
        """A detector that returns None (transient miss) must fall back to
        ground-truth data.xpos, matching today's exact pre-Task-4 behavior."""
        task, m, d, manip, real_cube_pos = self._build_task(
            cfg, cube_detector=_NoneDetector()
        )

        assert not task._grasp_confirmed

        xy = task.target_xy()

        assert np.allclose(xy, np.array(real_cube_pos[:2]))

    def test_default_cube_detector_none_matches_pre_task4_behavior(self, cfg):
        """Default construction (cube_detector parameter omitted entirely)
        must behave identically to before this change: ground truth used
        directly, with zero effect from the new optional parameter."""
        task, m, d, manip, real_cube_pos = self._build_task(cfg, _omit_param=True)

        assert task._cube_detector is None
        assert not task._grasp_confirmed

        xy = task.target_xy()

        assert np.allclose(xy, np.array(real_cube_pos[:2]))


class TestCubePosFreezeAtManipulationEntry:
    """Reliability fix: once seed_approach() runs (the coordinator's single
    WALKING->MANIPULATING entry hook), target_xy() must stop re-deriving the
    cube position from the detector on every call, holding the frozen value
    steady through APPROACHING/DESCENDING/GRASPING instead. This is what
    stops live camera frame-to-frame jitter from turning _wp_approach/
    _wp_descend into a continuously-moving target the EE can never converge
    on (see .superpowers/sdd/vision-task-6-report.md for the diagnosed bug
    this fixes)."""

    def _build_task(self, cfg, cube_detector):
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)

        from controllers.manipulation import ManipulationController  # noqa: PLC0415

        manip = ManipulationController(m, d)
        ee_home = manip.ee_position().copy()

        hover_z = 0.15
        ftp_offset = manip._ee_spec.ftp_offset
        cube_pos = [float(ee_home[0]), float(ee_home[1]), float(ee_home[2] - hover_z)]
        target_pos = [float(ee_home[0]) + 0.1, float(ee_home[1]), float(ee_home[2] - hover_z) + 0.006]

        task_cfg = dict(cfg["task"])
        task = PickAndPlaceTask(
            cube_pos, target_pos, m, d, manip, ftp_offset, task_cfg,
            cube_detector=cube_detector,
        )
        return task

    def test_unfrozen_before_seed_approach_tracks_live_detector(self, cfg):
        """Before seed_approach() is ever called (i.e. during WALKING),
        target_xy() must keep tracking the detector's live value -- proving
        WALKING's existing continuous-refresh behavior is untouched."""
        stub = _StubDetector([0.50, -0.20, 0.30])
        task = self._build_task(cfg, cube_detector=stub)

        assert task._cube_pos_frozen is False

        xy1 = task.target_xy()
        assert np.allclose(xy1, [0.50, -0.20])
        assert stub.calls == 1

        stub._fixed_pos = np.array([0.60, 0.10, 0.30])
        xy2 = task.target_xy()
        assert np.allclose(xy2, [0.60, 0.10])
        assert stub.calls == 2

    def test_target_xy_returns_frozen_value_during_approaching_despite_detector_change(self, cfg):
        """seed_approach() freezes whatever target_xy() last reported; later
        detector reads (simulating live camera jitter) must NOT change
        target_xy()'s return value or trigger any further detector calls."""
        stub = _StubDetector([0.50, -0.20, 0.30])
        task = self._build_task(cfg, cube_detector=stub)

        frozen_xy = task.target_xy().copy()   # WALKING's last refresh
        assert stub.calls == 1

        task.seed_approach(t=12.0)
        assert task._cube_pos_frozen is True

        # Simulate camera jitter: the detector now reports a different
        # position on every subsequent call.
        stub._fixed_pos = np.array([0.55, -0.15, 0.30])

        for _ in range(3):
            xy = task.target_xy()
            assert np.allclose(xy, frozen_xy), (
                "target_xy() must keep returning the frozen value, not "
                "chase the detector's now-changed reading"
            )

        assert stub.calls == 1, (
            "_refresh_cube_pos() must not be called again through "
            "target_xy() once frozen"
        )

    def test_releasing_resets_frozen_flag(self, cfg):
        """Defensive symmetry check: RELEASING resets _cube_pos_frozen back
        to False alongside the existing _grasp_confirmed reset."""
        stub = _StubDetector([0.50, -0.20, 0.30])
        task = self._build_task(cfg, cube_detector=stub)
        task.seed_approach(t=0.0)
        assert task._cube_pos_frozen is True

        task._set_phase(PickAndPlaceTask.Phase.RELEASING, 30.0)
        task.manip_step(coordinator=None, t=30.1, dt=0.005)

        assert task._cube_pos_frozen is False
