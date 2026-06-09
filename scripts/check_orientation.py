"""Check Go2 forward direction and joint axis conventions."""
import sys
from pathlib import Path
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "combined.xml")
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)

print("=== Shoulder positions (to determine front direction) ===")
for name in ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    bid = m.jnt_bodyid[jid]
    print(f"  {name}: xpos={d.xpos[bid].round(3)}")

print("\n=== Body positions after keyframe reset ===")
for name in ["base_link", "panda_link0"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        print(f"  {name}: xpos={d.xpos[bid].round(3)}")

print("\n=== Joint axis directions in world frame ===")
for name in ["FR_thigh_joint"]:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    bid = m.jnt_bodyid[jid]
    # Joint axis in local body frame
    axis_local = m.jnt_axis[jid]
    print(f"  {name}: local axis={axis_local.round(3)}")
    # Rotate axis to world frame using body orientation
    xquat = d.xquat[bid]
    # Convert quaternion to rotation matrix
    rot = np.zeros((3, 3))
    mujoco.mju_quat2Mat(rot.ravel(), xquat)
    axis_world = rot @ axis_local
    print(f"  {name}: world axis={axis_world.round(3)}")

# Now simulate one gait step to see which direction the thigh moves
print("\n=== Thigh direction test ===")
# Set thigh_joint angle slightly positive
jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "FR_thigh_joint")
qadr = m.jnt_qposadr[jid]
bid_thigh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FR_thigh")
bid_calf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")

# At stand pose
d2 = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d2, kid)
mujoco.mj_forward(m, d2)
thigh_pos_stand = d2.xpos[bid_thigh if bid_thigh >= 0 else 0].copy()

# Try FR foot body
bid_fr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FR_foot")
if bid_fr < 0:
    # Try alternate names
    for name in ["FR_foot_fixed", "FR_calf"]:
        bid_fr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid_fr >= 0:
            print(f"  FR end body: {name}")
            break

foot_pos_stand = d2.xpos[bid_fr].copy() if bid_fr >= 0 else None

# Add positive thigh angle
d2.qpos[qadr] += 0.3
mujoco.mj_forward(m, d2)
thigh_pos_plus = d2.xpos[bid_thigh if bid_thigh >= 0 else 0].copy()
foot_pos_plus = d2.xpos[bid_fr].copy() if bid_fr >= 0 else None

print(f"  Stand qpos[thigh]={d.qpos[qadr]:.3f} rad")
if foot_pos_stand is not None and foot_pos_plus is not None:
    delta = foot_pos_plus - foot_pos_stand
    print(f"  Adding +0.3 rad to FR_thigh: foot moves {delta.round(3)}")
    print(f"  -> +thigh moves foot in {'FORWARD (+x)' if delta[0] > 0 else 'BACKWARD (-x)'} direction")
