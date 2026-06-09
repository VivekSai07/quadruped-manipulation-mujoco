# Go2 + Franka Loco-Manipulation

A MuJoCo simulation of a Unitree Go2 quadruped with a Franka Panda arm mounted on its back, performing a full autonomous pick-and-place task — no external SDKs, no ROS, pure Python.

The robot walks to a table, lowers its stance to optimise arm workspace, reaches down to grasp a cube with the Panda arm, transports it to a placement plate, and returns the arm to home. The complete task runs in ~34 seconds of simulated time.

---

## Architecture

```
scripts/run_simulation.py   ← entry point (viewer / headless / record)
scripts/build_model.py      ← generates models/combined.xml from scratch
tasks/reach_task.py         ← logging wrapper around the coordinator
controllers/
  coordinator.py            ← 15-state task state machine
  locomotion.py             ← Go2 PD stand + sinusoidal trot gait
  manipulation.py           ← Panda velocity-IK + gripper control
configs/default.yaml        ← all tunable parameters
models/combined.xml         ← built MJCF (auto-generated, not hand-edited)
```

### State Machine

```
INIT → STANDING → WALKING → STOPPING → STABILIZING
     → ADJUSTING_HEIGHT → APPROACHING → DESCENDING → GRASPING
     → LIFTING → TRANSPORTING → LOWERING → RELEASING → RETURNING_HOME → DONE
```

Key engineering decisions:

| Problem | Solution |
|---|---|
| Cube slips during transport | Full 6-DOF kinematic attachment: position + quaternion locked to EE frame at grasp time, velocity zeroed every step |
| Jerky arm motion between states | Velocity IK (one Jacobian step per physics timestep) instead of periodic batch IK; intermediate Cartesian target interpolated at 10 cm/s |
| Go2 too tall for arm workspace | Adaptive height: `crouch_alpha` computed from horizontal reach and vertical reach below base, blends stand/crouch poses without hardcoded thresholds |
| Arm stuck at home after placing | `RETURNING_HOME` state advances a commanded joint target at 1.5 rad/s (independent of PD lag) until arm reaches home pose |

---

## Quick Start

### 1. Install dependencies

```bash
conda activate base
pip install mujoco numpy pyyaml opencv-python
```

### 2. Build the combined model

```bash
python scripts/build_model.py
```

### 3. Run

```bash
# Interactive viewer (real-time, pauseable)
python scripts/run_simulation.py

# Headless (fast, prints state transitions)
python scripts/run_simulation.py --no-viewer

# Record to MP4 (headless + video output)
python scripts/run_simulation.py --record

# Record with custom path and resolution
python scripts/run_simulation.py --record --video-path demo.mp4 --record-width 1920 --record-height 1080
```

> Note: do not pass `--duration 30` when recording — the task takes ~34 s. Omit `--duration` to use the config default (150 s); the simulation stops automatically when `DONE` is reached.

### 4. Run tests

```bash
pytest tests/ -v
```

39 tests covering model integrity, controller math, stability, and full task integration.

---

## Configuration

All parameters live in [configs/default.yaml](configs/default.yaml). Key knobs:

```yaml
simulation:
  max_duration: 150.0        # seconds before timeout

task:
  cube_pos:   [1.6, 0.0,  0.325]   # pickup cube world position
  target_pos: [1.6, 0.20, 0.331]   # placement plate center
  stop_distance: 0.65               # meters XY before Go2 stops walking
  grasp_hold_duration: 3.0          # seconds gripper holds closed before lift
  height_settle_time: 2.0           # seconds for Go2 to settle at new crouch
```

---

## Project Structure

```
Go2+FR/
├── assets/
│   ├── go2/            Unitree Go2 meshes (OBJ)
│   └── panda/          Franka Panda meshes (OBJ + STL)
├── configs/
│   └── default.yaml
├── controllers/
│   ├── base.py         BaseController ABC
│   ├── locomotion.py   Go2 PD + trot gait + crouch blend
│   ├── manipulation.py Panda velocity-IK, batch IK, gripper
│   └── coordinator.py  Task state machine (15 states)
├── models/
│   └── combined.xml    Auto-generated MJCF (git-ignored if large)
├── scripts/
│   ├── build_model.py  MJCF generator
│   ├── run_simulation.py  Main entry point
│   └── smoke_test*.py  Quick sanity scripts
├── tasks/
│   └── reach_task.py   Task wrapper with logging
├── tests/
│   ├── test_model.py
│   ├── test_controllers.py
│   ├── test_stability.py
│   └── test_task.py
├── archieve/           Iterative development history (m01–m14)
│   └── controllers/    ik_controller_m0/m1/m2, grasp_controller, etc.
└── configs/
    └── default.yaml
```

---

## Technical Details

### Robot Model

- **Go2**: 12-DOF quadruped (4 legs × 3 joints). Free joint for base. Leg motors: hip/thigh ±60 Nm, knee ±90 Nm (boosted from stock for arm payload).
- **Panda**: 7-DOF arm rigidly mounted 10 cm above Go2 base_link. Masses scaled to 35% of original (~6.5 kg from 18.5 kg) to match real payload capacity. Integrated PD actuators (`general` type with `gainprm`/`biasprm`).
- **Combined model**: 35 qpos (7 base + 12 legs + 7 arm + 2 fingers + 7 cube), 20 actuators, ~21 kg total mass.

### Velocity IK

Each physics step (dt = 0.005 s) the arm controller computes:

```
v_des = [Kp * pos_err,  Kr * rot_err]
dq    = J^T (J J^T + λ²I)^{-1} v_des
q_target += dq * dt          (integrated from commanded target, not measured)
```

This produces continuous smooth motion vs. the 3 cm "lurch" pattern of periodic batch IK.

### 6-DOF Kinematic Attachment

At grasp time:
```python
_grasp_R_local = ee_xmat.T @ cube_xmat   # cube rotation in EE frame
_grasp_offset  = cube_pos - ee_pos       # cube center offset from ee_site
```

Every step during LIFTING / TRANSPORTING / LOWERING:
```python
new_cube_R    = ee_xmat @ _grasp_R_local
new_cube_quat = mat2quat(new_cube_R)     # Shepperd's method
data.qpos[cube_adr:cube_adr+7] = [new_pos, new_quat]
data.qvel[cube_vel_adr:+6]     = 0.0
```

This prevents the cube tumbling or sliding during transport.

### Adaptive Height

```python
alpha_h = max(0, (horizontal_reach - 0.30) / 0.60)  # 0 at 30cm, 1 at 90cm
alpha_v = max(0, (vertical_reach   - 0.00) / 0.30)  # 0 at base level
alpha   = min(0.5, alpha_h * 0.35 + alpha_v * 0.25)
loco.set_crouch_alpha(alpha)  # blends _STAND_POSE and _CROUCH_POSE
```

For the default scenario (cube 64 cm away, 5 cm below base): `alpha ≈ 0.24` → ~4 cm lower stance.

### Physics Settings

```xml
<option timestep="0.005" cone="elliptic" impratio="100"
        integrator="implicitfast" iterations="50" tolerance="1e-10"/>
```

- `impratio="100"` stiffens contact and reduces cube sliding.
- `implicitfast` integrator is stable at 5 ms timestep with arm dynamics.
- Cube contact: `solref="0.002 1" solimp="0.9 0.95 0.001"` (stiff, high restitution damping).
- Finger tip contact geoms: `friction="1.5 0.05 0.01"` (high friction for reliable grasp).

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `mujoco` | ≥ 3.0 | Physics simulation + rendering |
| `numpy` | any | Math |
| `pyyaml` | any | Config loading |
| `opencv-python` | any | MP4 video encoding (recording only) |
| `pytest` | any | Test suite |

No ROS. No Unitree SDK. No external motion planners.

---

## Archive

The `archieve/` directory contains the full development lineage of the Franka Panda standalone simulation (before Go2 integration):

- `m01` — basic position IK
- `m02` — nullspace / velocity IK (the pattern used in this project)
- `m03` — trajectory planning
- `m04` — sense-plan-recover
- `m05` — RRT path planning
- `m06` — keyboard teleoperation
- `m07` — machine vision integration
- `m08` — autonomous obstacle avoidance
- `m09` — YOLO object detection
- `m10` — MoCap teleoperation
- `m11` — VLM-guided pick-and-place
- `m12` — Florence-2 integration
- `m13` — reactive manipulation
- `m14` — TAMP (Task and Motion Planning)
