"""
Main entry point for Go2+Panda loco-manipulation simulation.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --config configs/default.yaml
    python scripts/run_simulation.py --no-viewer        (headless, no output)
    python scripts/run_simulation.py --record           (headless, saves simulation_recording.mp4)
    python scripts/run_simulation.py --record --duration 30

The script:
  1. Loads the combined MJCF model
  2. Resets to keyframe "home"
  3. Runs the ReachTask controller
  4. Displays via MuJoCo passive viewer, runs headless, or records to MP4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Force line-buffered stdout so output appears through conda run on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import mujoco
import numpy as np
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.reach_task import ReachTask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Go2+Panda loco-manipulation demo")
    p.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    p.add_argument("--no-viewer", action="store_true", help="Run headless (no GUI, no video)")
    p.add_argument("--record", action="store_true",
                   help="Record headless simulation to --video-path (overwrites each run)")
    p.add_argument("--video-path", default="simulation_recording.mp4",
                   help="Output video file (default: simulation_recording.mp4)")
    p.add_argument("--record-fps", type=int, default=30,
                   help="Video frame rate (default: 30)")
    p.add_argument("--record-width", type=int, default=1280,
                   help="Video width in pixels (default: 1280)")
    p.add_argument("--record-height", type=int, default=720,
                   help="Video height in pixels (default: 720)")
    p.add_argument("--duration", type=float, default=None, help="Override max duration (s)")
    p.add_argument("--build-model", action="store_true",
                   help="Rebuild combined.xml before running")
    return p.parse_args()


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


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

    Always overwrites video_path so repeated runs produce a single file.
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

    # Optionally rebuild model XML
    if args.build_model:
        print("Rebuilding model...")
        from scripts.build_model import main as build_main  # noqa: PLC0415
        build_main()

    model_path = cfg["simulation"]["model_path"]
    max_duration = args.duration or cfg["simulation"].get("max_duration", 45.0)

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

    task = ReachTask(model, data, cfg)

    print(f"\n{'='*60}")
    print("Go2 + Franka Panda Loco-Manipulation Demo")
    print(f"  Model: nq={model.nq}, nu={model.nu}, nbody={model.nbody}")
    print(f"  Total mass: {sum(model.body_mass):.2f} kg")
    print(f"  Task: Walk to cube at {cfg['task']['cube_pos']}, reach with arm")
    print(f"  Max duration: {max_duration:.1f}s")
    print("  States: INIT->STANDING->WALKING->STOPPING->STABILIZING->ADJUSTING_HEIGHT->APPROACHING->DESCENDING->GRASPING->LIFTING->TRANSPORTING->LOWERING->RELEASING->RETURNING_HOME->DONE")
    print(f"{'='*60}\n")

    if args.record:
        success = run_recorded(
            model, data, task, max_duration, cfg,
            fps=args.record_fps,
            width=args.record_width,
            height=args.record_height,
            video_path=args.video_path,
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
