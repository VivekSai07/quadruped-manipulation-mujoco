"""
Debug script: run to GRASPING state and print detailed finger/cube geometry.
Helps diagnose why physical contact is not occurring.
"""
import sys
import yaml
import mujoco
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from controllers.coordinator import TaskCoordinator, TaskState
from tasks.reach_task import ReachTask

cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "combined.xml"))
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, kid)

task = ReachTask(m, d, cfg)
dt = m.opt.timestep
coord = task.coordinator

# Body / site IDs
ee_sid   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
lf_bid   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_left_finger")
rf_bid   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_right_finger")
hand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")

print(f"ee_site id={ee_sid}, cube_body id={cube_bid}, lf_body id={lf_bid}, rf_body id={rf_bid}")

prev_state = None
grasping_printed = False

for i in range(int(50.0 / dt)):
    task.step(dt)
    mujoco.mj_step(m, d)
    task.coordinator.post_physics_step()
    t = float(d.time)

    state = coord.state

    # Print geometry snapshot when we enter GRASPING
    if state == TaskState.GRASPING and not grasping_printed:
        grasping_printed = True
        mujoco.mj_fwdPosition(m, d)
        ee_pos   = d.site_xpos[ee_sid].copy()
        cube_pos = d.xpos[cube_bid].copy()
        lf_pos   = d.xpos[lf_bid].copy()
        rf_pos   = d.xpos[rf_bid].copy()
        hand_pos = d.xpos[hand_bid].copy()

        # Finger joint angles
        fj1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        fj2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        q_fj1 = d.qpos[m.jnt_qposadr[fj1_id]]
        q_fj2 = d.qpos[m.jnt_qposadr[fj2_id]]

        print(f"\n=== GRASPING state entered at t={t:.3f}s ===")
        print(f"  EE site   : {ee_pos}")
        print(f"  Cube pos  : {cube_pos}")
        print(f"  Hand pos  : {hand_pos}")
        print(f"  L finger  : {lf_pos}")
        print(f"  R finger  : {rf_pos}")
        print(f"  Finger q  : fj1={q_fj1:.4f}m  fj2={q_fj2:.4f}m  (max=0.04m)")
        print(f"  EE->cube   : {np.linalg.norm(ee_pos - cube_pos):.4f}m")
        print(f"  Lf->cube   : {np.linalg.norm(lf_pos - cube_pos):.4f}m")
        print(f"  Rf->cube   : {np.linalg.norm(rf_pos - cube_pos):.4f}m")
        print(f"  ncon={d.ncon}")
        for ci in range(d.ncon):
            c = d.contact[ci]
            b1 = int(m.geom_bodyid[c.geom1])
            b2 = int(m.geom_bodyid[c.geom2])
            g1n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
            g2n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
            b1n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or f"b{b1}"
            b2n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or f"b{b2}"
            print(f"  contact {ci}: [{b1n}:{g1n}] vs [{b2n}:{g2n}]")

    # Print once per second during grasping
    if state == TaskState.GRASPING and i % 200 == 0:
        mujoco.mj_fwdPosition(m, d)
        ee_pos = d.site_xpos[ee_sid].copy()
        cube_pos = d.xpos[cube_bid].copy()
        fj1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        q_fj1 = d.qpos[m.jnt_qposadr[fj1_id]]
        grasped = coord.manip.is_grasped()
        print(f"  t={t:.2f}s GRASPING: ee={ee_pos.round(3)} cube={cube_pos.round(3)} fj={q_fj1:.4f} grasped={grasped} ncon={d.ncon}")

    if state == TaskState.LIFTING and i % 200 == 0:
        mujoco.mj_fwdPosition(m, d)
        cube_pos = d.xpos[cube_bid].copy()
        ee_pos = d.site_xpos[ee_sid].copy()
        fj1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        q_fj1 = d.qpos[m.jnt_qposadr[fj1_id]]
        grasped = coord.manip.is_grasped()
        print(f"  t={t:.2f}s LIFTING: cube_z={cube_pos[2]:.4f} ee_z={ee_pos[2]:.4f} fj={q_fj1:.4f} grasped={grasped} ncon={d.ncon}")

    if state == TaskState.TRANSPORTING or coord.is_done:
        mujoco.mj_fwdPosition(m, d)
        cube_pos = d.xpos[cube_bid].copy()
        print(f"\n  Reached {state.value} at t={t:.2f}s, cube_z={cube_pos[2]:.4f}m")
        break

print("\nDone.")
