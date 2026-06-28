"""
Main entry point for Go2 + swappable-arm loco-manipulation simulation.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --config configs/default.yaml
    python scripts/run_simulation.py --no-viewer        (headless, no output)
    python scripts/run_simulation.py --record           (headless, auto-named video in media/)
    python scripts/run_simulation.py --record --duration 30
    python scripts/run_simulation.py --arm kinova_gen3

The script:
  1. Loads the combined MJCF model (rebuilding it if the cached model was
     built for a different arm/end-effector combo)
  2. Resets to keyframe "home"
  3. Runs the ReachTask controller
  4. Displays via MuJoCo passive viewer, runs headless, or records to MP4
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Force line-buffered stdout so output appears through conda run on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import mujoco
import numpy as np
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from controllers.arms import ARMS, DEFAULT_ARM, get_arm_spec, validate_combo
from controllers.end_effectors import DEFAULT_END_EFFECTOR, END_EFFECTORS, get_spec
from tasks.base import Task
from tasks.pick_and_place import PickAndPlaceTask
from tasks.push import PushTask
from tasks.reach_only import ReachOnlyTask
from tasks.reach_task import ReachTask

if TYPE_CHECKING:
    from controllers.manipulation import ManipulationController

_ARM_STAMP_RE = re.compile(r"ARM_STAMP:\s*(\S+)")
_EE_STAMP_RE = re.compile(r"END_EFFECTOR_STAMP:\s*(\S+)")


def _build_task(
    task_item_cfg: dict[str, Any],
    manip: "ManipulationController",
    ftp_offset: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> Task:
    """Construct a single Task from one task-config item (either the flat
    cfg["task"] dict, or one entry of cfg["task"]["sequence"]), dispatching
    on `type` (default "pick_and_place" -- today's only behavior).

    Reuses existing config keys rather than inventing new ones:
      - pick_and_place: cube_pos / target_pos (today's exact keys).
      - reach_only: reuses cube_pos as the literal reach target (no separate
        "target_point" key is introduced -- ReachOnlyTask just needs *some*
        point, and cube_pos is already the point of interest in any task
        config that didn't bother to add a push/pick-specific key).
      - push: reuses cube_pos / target_pos as object_pos / target_pos,
        mirroring PickAndPlaceTask's own convention.
    """
    task_type = task_item_cfg.get("type", "pick_and_place")
    cube_pos = task_item_cfg.get("cube_pos", [1.6, 0.0, 0.325])
    target_pos = task_item_cfg.get("target_pos", [1.6, 0.20, 0.331])

    if task_type == "pick_and_place":
        return PickAndPlaceTask(cube_pos, target_pos, model, data, manip, ftp_offset, task_item_cfg)
    elif task_type == "reach_only":
        return ReachOnlyTask(cube_pos, model, data, manip, task_item_cfg)
    elif task_type == "push":
        return PushTask(cube_pos, target_pos, model, data, manip, ftp_offset, task_item_cfg)
    else:
        raise ValueError(f"Unknown task.type: {task_type!r}")


def _make_task_factory(model: mujoco.MjModel, data: mujoco.MjData, cfg: dict[str, Any]):
    """Build the actual TaskFactory callable for ReachTask, given model/data
    (available at call time in main()) and the full top-level config."""
    task_cfg = cfg.get("task", {})
    sequence = task_cfg.get("sequence")

    def factory(manip: "ManipulationController", ftp_offset: float, _task_cfg: dict[str, Any]) -> Task:
        if sequence:
            items = [_build_task(item, manip, ftp_offset, model, data) for item in sequence]
            for a, b in zip(items, items[1:]):
                a.set_next_task(b)
            return items[0]
        return _build_task(task_cfg, manip, ftp_offset, model, data)

    return factory


def _describe_task(cfg: dict[str, Any]) -> str:
    """Human-readable one-line description of the active task(s) for the
    startup banner, replacing the old pick-and-place-only hardcoded line."""
    task_cfg = cfg.get("task", {})
    sequence = task_cfg.get("sequence")
    if sequence:
        kinds = ", ".join(item.get("type", "pick_and_place") for item in sequence)
        return f"Sequence of {len(sequence)} task(s): {kinds}"

    task_type = task_cfg.get("type", "pick_and_place")
    cube_pos = task_cfg.get("cube_pos", [1.6, 0.0, 0.325])
    target_pos = task_cfg.get("target_pos", [1.6, 0.20, 0.331])
    if task_type == "pick_and_place":
        return f"Walk to cube at {cube_pos}, grasp, transport to {target_pos}"
    elif task_type == "reach_only":
        return f"Walk to point {cube_pos}, reach with arm (no grasp)"
    elif task_type == "push":
        return f"Walk to object at {cube_pos}, push toward {target_pos}"
    return f"Unknown task type {task_type!r}"


def _model_stamps(model_path: str) -> tuple[str | None, str | None]:
    """Return the (arm, end_effector) names baked into models/combined.xml,
    or (None, None) if the file is missing or has no stamp comments."""
    path = Path(model_path)
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    arm_match = _ARM_STAMP_RE.search(text)
    ee_match = _EE_STAMP_RE.search(text)
    return (
        arm_match.group(1) if arm_match else None,
        ee_match.group(1) if ee_match else None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Go2+Panda loco-manipulation demo")
    p.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    p.add_argument("--no-viewer", action="store_true", help="Run headless (no GUI, no video)")
    p.add_argument("--record", action="store_true",
                   help="Record headless simulation to --video-path (auto-named if omitted)")
    p.add_argument("--video-path", default=None,
                   help="Output video file (default: auto-named media/simulation_recording_<arm>_<end-effector>.mp4, "
                        "non-clobbering)")
    p.add_argument("--record-fps", type=int, default=30,
                   help="Video frame rate (default: 30)")
    p.add_argument("--record-width", type=int, default=1280,
                   help="Video width in pixels (default: 1280)")
    p.add_argument("--record-height", type=int, default=720,
                   help="Video height in pixels (default: 720)")
    p.add_argument("--duration", type=float, default=None, help="Override max duration (s)")
    p.add_argument("--build-model", action="store_true",
                   help="Rebuild combined.xml before running")
    p.add_argument("--arm", choices=sorted(ARMS), default=DEFAULT_ARM,
                   help=f"Arm to mount on the Go2 trunk (default: {DEFAULT_ARM})")
    p.add_argument("--end-effector", choices=sorted(END_EFFECTORS), default=None,
                   help="End-effector to mount on the arm's wrist "
                        "(default: the chosen arm's default end-effector)")
    p.add_argument("--cube-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Override the cube's spawn position (relocates the physical cube "
                        "body, not just config -- the model XML hardcodes its default "
                        "position). Useful for demonstrating off-axis WALKING/turning; the "
                        "cube may be off the worktable so the later pick-and-place phases "
                        "are not guaranteed to complete.")
    return p.parse_args()


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve_video_path(requested: str | None, arm: str, end_effector: str) -> str:
    """Return the video output path: passed through verbatim if the user
    requested one explicitly, otherwise auto-named from arm/end-effector and
    never overwriting an existing file (increments a numeric suffix)."""
    if requested is not None:
        return requested
    media_dir = Path("media")
    media_dir.mkdir(parents=True, exist_ok=True)
    stem = f"simulation_recording_{arm}_{end_effector}"
    candidate = media_dir / f"{stem}.mp4"
    n = 2
    while candidate.exists():
        candidate = media_dir / f"{stem}_{n}.mp4"
        n += 1
    return str(candidate)


def setup_viewer(model: mujoco.MjModel, data: mujoco.MjData, cfg: dict) -> Any:
    """Configure passive viewer camera."""
    import mujoco.viewer  # noqa: PLC0415
    viewer = mujoco.viewer.launch_passive(model, data)
    vcfg = cfg.get("viewer", {})
    viewer.cam.azimuth = vcfg.get("camera_azimuth", -140.0)
    viewer.cam.elevation = vcfg.get("camera_elevation", -20.0)
    viewer.cam.distance = vcfg.get("camera_distance", 3.5)
    viewer.cam.lookat[:] = [0.5, 0.0, 0.3]
    return viewer


def _make_camera(cfg: dict) -> mujoco.MjvCamera:
    """Build a free MjvCamera from viewer config (used by the recorder)."""
    vcfg = cfg.get("viewer", {})
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = vcfg.get("camera_azimuth", -140.0)
    cam.elevation = vcfg.get("camera_elevation", -20.0)
    cam.distance = vcfg.get("camera_distance", 3.5)
    cam.lookat[:] = [0.5, 0.0, 0.3]
    return cam


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task: ReachTask,
    max_duration: float,
) -> bool:
    """Headless loop for testing without a display."""
    dt = model.opt.timestep
    n_steps = int(max_duration / dt)
    print(f"Running headless for {max_duration:.1f}s ({n_steps} steps)...")
    for _ in range(n_steps):
        task.step(dt)
        mujoco.mj_step(model, data)
        task.coordinator.post_physics_step()   # snap cube after constraint solve
        if task.is_done:
            break
    return task.is_done


def run_recorded(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task: ReachTask,
    max_duration: float,
    cfg: dict,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    video_path: str = "simulation_recording.mp4",
) -> bool:
    """Run headless and encode every frame to an MP4 via OpenCV.

    Always overwrites video_path as given -- non-clobbering auto-naming
    (when the caller didn't request a specific path) is the caller's
    responsibility, via _resolve_video_path().
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not installed. Run: pip install opencv-python")
        return False

    dt = model.opt.timestep
    n_steps = int(max_duration / dt)
    # Record one frame every this many simulation steps to hit the target fps.
    record_every = max(1, round(1.0 / (fps * dt)))
    actual_fps = 1.0 / (record_every * dt)

    cam = _make_camera(cfg)
    renderer = mujoco.Renderer(model, height=height, width=width)

    # MP4V produces .mp4 files that play everywhere; use avc1/H.264 when available.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, actual_fps, (width, height))
    if not out.isOpened():
        print(f"ERROR: Could not open video writer for {video_path!r}")
        renderer.close()
        return False

    print(f"Recording to {video_path!r}  ({width}x{height} @ {actual_fps:.1f} fps)")
    print(f"  Simulation: {max_duration:.1f}s ({n_steps} steps, 1 frame / {record_every} steps)")

    frame_count = 0
    for step in range(n_steps):
        task.step(dt)
        mujoco.mj_step(model, data)
        task.coordinator.post_physics_step()   # snap cube after constraint solve

        if step % record_every == 0:
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()                        # (H, W, 3) uint8 RGB
            bgr = rgb[:, :, ::-1]                         # OpenCV expects BGR
            out.write(bgr)
            frame_count += 1

        if task.is_done:
            break

    out.release()
    renderer.close()
    print(f"  Wrote {frame_count} frames -> {video_path!r}")
    return task.is_done


def run_with_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task: ReachTask,
    max_duration: float,
    cfg: dict,
) -> bool:
    """Interactive viewer loop."""
    import mujoco.viewer  # noqa: PLC0415

    dt = model.opt.timestep
    viewer = setup_viewer(model, data, cfg)

    with viewer:
        while viewer.is_running() and float(data.time) < max_duration:
            step_start = time.perf_counter()
            task.step(dt)
            mujoco.mj_step(model, data)
            task.coordinator.post_physics_step()   # snap cube after constraint solve
            viewer.sync()
            # Real-time pacing
            elapsed = time.perf_counter() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    return task.is_done


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    model_path = cfg["simulation"]["model_path"]
    max_duration = args.duration or cfg["simulation"].get("max_duration", 45.0)

    effective_ee = args.end_effector or get_arm_spec(args.arm).default_end_effector
    validate_combo(args.arm, effective_ee)

    # Rebuild model XML if explicitly requested, or if the cached model was
    # built for a different arm/end-effector combo (or doesn't exist / has
    # no stamps).
    current_arm_stamp, current_ee_stamp = _model_stamps(model_path)
    if args.build_model or current_arm_stamp != args.arm or current_ee_stamp != effective_ee:
        print(f"Rebuilding model for arm '{args.arm}' + end-effector '{effective_ee}'...")
        from scripts.build_model import main as build_main  # noqa: PLC0415
        build_main(arm=args.arm, end_effector=effective_ee)

    print(f"Loading model: {model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        print("Run: python scripts/build_model.py  to rebuild combined.xml")
        return 1

    data = mujoco.MjData(model)

    # Reset to home keyframe
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        print(f"Reset to keyframe 'home'")
    else:
        print("WARNING: 'home' keyframe not found -- using default pose")

    if args.cube_pos is not None:
        cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qadr = int(model.jnt_qposadr[cube_jid])
        data.qpos[cube_qadr:cube_qadr + 3] = args.cube_pos
        mujoco.mj_forward(model, data)
        cfg["task"]["cube_pos"] = list(args.cube_pos)
        print(f"Relocated cube to {args.cube_pos}")

    # Only pass a task_factory when the config actually opts into
    # task-type selection or sequencing -- when task.type/task.sequence are
    # both absent (today's default.yaml), task_factory stays None and
    # ReachTask/TaskCoordinator take their original, unmodified default-
    # construction path (a hardcoded PickAndPlaceTask), guaranteeing
    # byte-for-byte identical behavior to before this change.
    task_item_cfg = cfg.get("task", {})
    if "type" in task_item_cfg or "sequence" in task_item_cfg:
        task_factory = _make_task_factory(model, data, cfg)
    else:
        task_factory = None

    task = ReachTask(model, data, cfg, arm=args.arm, end_effector=effective_ee, task_factory=task_factory)

    print(f"\n{'='*60}")
    print("Go2 Loco-Manipulation Demo")
    print(f"  Arm: {get_arm_spec(args.arm).display_name}")
    print(f"  End-effector: {get_spec(effective_ee).display_name}")
    print(f"  Model: nq={model.nq}, nu={model.nu}, nbody={model.nbody}")
    print(f"  Total mass: {sum(model.body_mass):.2f} kg")
    print(f"  Task: {_describe_task(cfg)}")
    print(f"  Max duration: {max_duration:.1f}s")
    print("  States: INIT->STANDING->WALKING->STOPPING->STABILIZING->ADJUSTING_HEIGHT->MANIPULATING->RETURNING_HOME->DONE")
    print(f"{'='*60}\n")

    if args.record:
        video_path = _resolve_video_path(args.video_path, args.arm, effective_ee)
        success = run_recorded(
            model, data, task, max_duration, cfg,
            fps=args.record_fps,
            width=args.record_width,
            height=args.record_height,
            video_path=video_path,
        )
    elif args.no_viewer:
        success = run_headless(model, data, task, max_duration)
    else:
        try:
            success = run_with_viewer(model, data, task, max_duration, cfg)
        except Exception as e:
            print(f"Viewer error ({e}), falling back to headless")
            success = run_headless(model, data, task, max_duration)

    print(f"\n{'='*60}")
    if success:
        print(f"SUCCESS: Task completed at t={task.success_time:.2f}s")
    else:
        print(f"TIMEOUT: Task did not complete within {max_duration:.1f}s")
        print(f"  Final state: {task.coordinator.state.value}")
    print(f"{'='*60}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
