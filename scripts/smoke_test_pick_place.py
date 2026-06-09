"""
Headless smoke test for the full pick-and-place pipeline.
Run: conda run -n base python scripts/smoke_test_pick_place.py
"""
import sys
import yaml
import mujoco
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tasks.reach_task import ReachTask

cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "combined.xml"))
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, kid)

task = ReachTask(m, d, cfg)
dt = m.opt.timestep
max_t = cfg["simulation"]["max_duration"]
n_steps = int(max_t / dt)
last_print = -1.0

for i in range(n_steps):
    task.step(dt)
    mujoco.mj_step(m, d)
    t = float(d.time)
    if t - last_print >= 2.0:
        last_print = t
        print(task.coordinator.status_line(t))
    if task.is_done:
        break

print()
coord = task.coordinator
a = coord._cube_qpos_adr
cube_pos = d.qpos[a:a + 3].copy()
target = coord._target_pos
xy_err = float(np.linalg.norm(cube_pos[:2] - target[:2]))
z_err = float(abs(cube_pos[2] - target[2]))

print(f"Final state:    {coord.state.value}")
print(f"Cube pos:       [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}]")
print(f"Target pos:     [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
print(f"XY error:       {xy_err:.3f} m")
print(f"Z  error:       {z_err:.3f} m")
print(f"Placement OK:   {coord.placement_verified()}")
print(f"Sim time:       {float(d.time):.2f} s")
print()
if coord.state.value == "done":
    print("PIPELINE COMPLETE: Go2 walked → arm grasped → placed cube on plate")
else:
    print(f"Pipeline stopped in state: {coord.state.value}")
    print(f"Increase max_duration in config if still placing.")
