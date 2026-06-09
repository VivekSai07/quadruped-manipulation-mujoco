"""Quick smoke test: verify forward locomotion and full task pipeline."""
import sys
from pathlib import Path
import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from tasks.reach_task import ReachTask

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")
CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "default.yaml")

cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, kid)

task = ReachTask(model, data, cfg)
dt = model.opt.timestep
max_t = 45.0
n_steps = int(max_t / dt)
report_interval = int(1.0 / dt)

cube_pos = np.array(cfg["task"]["cube_pos"])
print(f"Cube at {cube_pos}")
print(f"{'t':>6s} | {'state':10s} | {'height':>7s} | {'base→cube':>10s} | {'ee→target':>10s}")
print("-" * 60)

prev_dist = None
motion_confirmed = False

for step in range(n_steps):
    task.step(dt)
    mujoco.mj_step(model, data)

    t = float(data.time)
    if step % report_interval == 0:
        status = task.coordinator.status_line(t)
        print(status)

        dist = task.coordinator._base_xy_distance_to_cube()
        state = task.coordinator.state.value
        if prev_dist is not None and state == "walking" and dist < prev_dist - 0.005:
            motion_confirmed = True
        prev_dist = dist

    if task.is_done:
        print(f"\nSUCCESS at t={t:.2f}s")
        break
else:
    print(f"\nTIMEOUT after {max_t:.1f}s | final state: {task.coordinator.state.value}")

print(f"\nForward motion confirmed: {motion_confirmed}")
