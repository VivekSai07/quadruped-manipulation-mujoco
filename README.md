# Go2 + Swappable Arm Loco-Manipulation

A MuJoCo simulation of a Unitree Go2 quadruped with a swappable robotic arm — Franka Panda or Kinova Gen3 — mounted on its back, performing a full autonomous pick-and-place task — no external SDKs, no ROS, pure Python.

The robot walks to a table, lowers its stance to optimise arm workspace, reaches down to grasp a cube with the arm, transports it to a placement plate, and returns the arm to home. The complete task runs in ~34 seconds of simulated time.

Both arms support the stock Franka two-finger gripper or a vendored Robotiq 2F-85 adaptive gripper (Kinova currently supports Robotiq only — see [Technical Details](#robot-model)).

## Demo

🎥 **Simulation Recording:**  
[Watch the demo video](https://github.com/VivekSai07/quadruped-manipulation-mujoco/blob/main/media/simulation_recording.mp4)

🎥 **Locomotion turning + full pick-and-place** (off-axis cube, see `--cube-pos` above):  
[Watch the turning demo](https://github.com/VivekSai07/quadruped-manipulation-mujoco/blob/main/media/locomotion_turning_complete.mp4)

---

## Architecture

```
scripts/run_simulation.py   ← entry point (viewer / headless / record)
scripts/build_model.py      ← generates models/combined.xml from scratch
tasks/
  base.py                   ← Task ABC: the pluggable-task strategy interface
  reach_task.py             ← logging wrapper around the coordinator
  pick_and_place.py         ← PickAndPlaceTask: walk-grasp-transport-place
  reach_only.py             ← ReachOnlyTask: walk-and-reach, no grasp
  push.py                   ← PushTask: walk-and-push via real contact physics
controllers/
  arms.py                   ← arm registry (Franka Panda, Kinova Gen3) + combo validation
  end_effectors.py          ← end-effector registry (Franka hand, Robotiq 2F-85) + mount overrides
  coordinator.py            ← generic outer task state machine
  locomotion.py             ← Go2 PD stand + sinusoidal trot gait
  manipulation.py           ← arm velocity-IK + gripper control (arm-agnostic)
configs/default.yaml        ← all tunable parameters
models/combined.xml         ← built MJCF (auto-generated, not hand-edited)
```

### Task Strategy Abstraction

`tasks/base.py` defines `Task`, an ABC that owns everything from "the robot has arrived" to "this object is done," while `TaskCoordinator` (`controllers/coordinator.py`) owns the generic parts -- locomotion, stabilization, adaptive height, and returning home -- independent of what manipulation behavior is running. Four concrete strategies exist today: `PickAndPlaceTask` (`tasks/pick_and_place.py`, the original grasp/transport/place behavior), `ReachOnlyTask` (`tasks/reach_only.py`, walk-and-reach with no grasp), and `PushTask` (`tasks/push.py`, walk-and-push via real contact friction instead of kinematic attachment). Tasks can be chained for multi-object sequencing via `task.set_next_task(next_task)` / `task.next_task()`: once a task finishes, `TaskCoordinator` automatically swaps to its chained successor and walks toward the next target instead of going `DONE`.

### State Machine

```
INIT → STANDING → WALKING → STOPPING → STABILIZING
     → ADJUSTING_HEIGHT → MANIPULATING → RETURNING_HOME → DONE
```

This outer machine is generic across all `Task` types -- `MANIPULATING` simply delegates to the active `Task`'s own `manip_step()` every physics step, and the task's internal sub-phases are invisible to the coordinator. `PickAndPlaceTask`'s 7 sub-phases (`APPROACHING → DESCENDING → GRASPING → LIFTING → TRANSPORTING → LOWERING → RELEASING`, with a `REGRASP` retry branch) are just one example: `ReachOnlyTask` has no sub-phases at all (a single reach, nothing to delegate to), and `PushTask` has 3 (`APPROACHING → DESCENDING → PUSHING`).

Key engineering decisions:

| Problem | Solution |
|---|---|
| Cube slips during transport | Full 6-DOF kinematic attachment: position + quaternion locked to EE frame at grasp time, velocity zeroed every step |
| Jerky arm motion between states | Velocity IK (one Jacobian step per physics timestep) instead of periodic batch IK; intermediate Cartesian target interpolated at 10 cm/s |
| Go2 too tall for arm workspace | Adaptive height: `crouch_alpha` computed from horizontal reach and vertical reach below base, blends stand/crouch poses without hardcoded thresholds |
| Arm stuck at home after placing | `RETURNING_HOME` state advances a commanded joint target at 1.5 rad/s (independent of PD lag) until arm reaches home pose |
| Go2 could only walk straight ahead | `WALKING` now computes bearing-to-cube every step and steers via differential left/right trot stride (`LocomotionController.set_heading()`), with speed ramped down only in the final approach (`set_speed_scale()`) |

---

## Quick Start

### 1. Install dependencies

```bash
conda activate base
pip install mujoco numpy pyyaml opencv-python
```

### 2. Build the combined model

```bash
python scripts/build_model.py                                    # default: Franka Panda + stock gripper
python scripts/build_model.py --arm kinova_gen3                   # Kinova Gen3 + Robotiq 2F-85 (its default)
python scripts/build_model.py --end-effector robotiq_2f85         # Franka Panda + Robotiq 2F-85
```

### 3. Run

```bash
# Interactive viewer (real-time, pauseable) -- defaults to Franka Panda
python scripts/run_simulation.py

# Headless (fast, prints state transitions)
python scripts/run_simulation.py --no-viewer

# Swap the arm -- model is rebuilt automatically if the cached one doesn't match
python scripts/run_simulation.py --arm kinova_gen3 --no-viewer

# Swap the end-effector (defaults to the arm's own default if omitted)
python scripts/run_simulation.py --end-effector robotiq_2f85 --no-viewer

# Record to MP4 (headless + video output, auto-named media/simulation_recording_<arm>_<end-effector>.mp4)
python scripts/run_simulation.py --record

# Record with a custom path (resolution capped at 1280x720, see note below)
python scripts/run_simulation.py --record --video-path demo.mp4 --record-width 1280 --record-height 720

# Force a rebuild even if the cached model already matches the requested combo
python scripts/run_simulation.py --build-model

# Demo the locomotion turning: relocate the cube off-axis (still on the
# worktable, so the full pick-and-place still completes) and record it
python scripts/run_simulation.py --cube-pos 1.6 0.32 0.325 --no-viewer --record --video-path media/locomotion_turning_complete.mp4 --duration 45
```

> Note: do not pass `--duration 30` when recording — the task takes ~34 s. Omit `--duration` to use the config default (150 s); the simulation stops automatically when `DONE` is reached.
>
> Note: `--record-width`/`--record-height` cannot exceed the model's offscreen framebuffer, set via `<visual><global offwidth="1280" offheight="720"/></visual>` in `scripts/build_model.py`. Requesting a larger resolution raises `ValueError: Image width ... > framebuffer width ...`. To record larger than 1280x720, bump `offwidth`/`offheight` in `build_model.py` and rebuild.
>

> Note: Kinova Gen3 only supports the Robotiq 2F-85 gripper — `--arm kinova_gen3 --end-effector franka` raises a `ValueError` before touching any files.
>
> Note: `--cube-pos` relocates the *physical* cube (the worktable/plate stay fixed, so keep the cube within roughly `x∈[1.36,1.84], y∈[-0.33,0.33]` relative to the table to stay on its surface — see [Robot Model](#robot-model)). Outside that range the cube falls to the ground and only the WALKING/turning portion is meaningful; the pick-and-place phases won't complete. A config-only `task.cube_pos` override does **not** move the cube (it's hardcoded in the compiled model) — `--cube-pos` is the only way to actually relocate it.

### 4. Run tests

```bash
pytest tests/ -v
```

105 tests covering model integrity, controller math, stability, full task integration, arm/end-effector variant combinations, CLI helpers, locomotion turning/speed control, REGRASP fault recovery, the `ReachOnlyTask`/`PushTask` task types, multi-object sequencing via `next_task()`, and config-driven task-type/sequence selection.

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

### Task type selection (`task.type` / `task.sequence`)

`scripts/run_simulation.py` reads an optional `task.type` key (default `pick_and_place`, so today's default config is unaffected) to choose which `Task` strategy to run, reusing the existing `cube_pos`/`target_pos` keys rather than introducing new ones per type:

```yaml
# Pick-and-place (default -- identical to omitting task.type entirely)
task:
  type: pick_and_place
  cube_pos:   [1.6, 0.0,  0.325]
  target_pos: [1.6, 0.20, 0.331]

# Reach-only: walk to a point and reach, no grasp. cube_pos is reused as
# the literal reach target -- no separate "target_point" key is introduced.
task:
  type: reach_only
  cube_pos: [1.6, 0.0, 0.45]

# Push: walk to an object and push it via real contact physics (no grasp,
# no kinematic attachment). cube_pos/target_pos are reused as object_pos/
# target_pos, mirroring pick_and_place's own convention.
task:
  type: push
  cube_pos:   [1.6, 0.20, 0.325]
  target_pos: [1.6, -0.20, 0.325]
```

For multi-object runs, `task.sequence` chains a list of per-item dicts (each with its own `type` plus that type's own fields) via `Task.set_next_task()` -- the coordinator automatically walks toward and runs each task in turn instead of going `DONE` after the first:

```yaml
task:
  sequence:
    - type: pick_and_place
      cube_pos:   [1.6, 0.0,  0.325]
      target_pos: [1.6, 0.20, 0.331]
    - type: push
      cube_pos:   [1.6, 0.20, 0.325]
      target_pos: [1.6, -0.20, 0.325]
```

`task.sequence` is opt-in only (absent from `configs/default.yaml`'s shipped defaults) -- add it to your own config dict or YAML file when you want it.

---

## Project Structure

```
Go2+FR/
├── assets/
│   ├── go2/             Unitree Go2 meshes (OBJ)
│   ├── panda/           Franka Panda meshes (OBJ + STL)
│   ├── kinova_gen3/     Kinova Gen3 meshes (vendored from upstream)
│   └── robotiq_2f85/    Robotiq 2F-85 gripper meshes
├── configs/
│   └── default.yaml
├── controllers/
│   ├── base.py          BaseController ABC
│   ├── arms.py           Arm registry (ArmSpec for Franka Panda / Kinova Gen3) + validate_combo
│   ├── end_effectors.py  End-effector registry (EndEffectorSpec, MountOverride) for Franka hand / Robotiq 2F-85
│   ├── locomotion.py    Go2 PD + trot gait + crouch blend
│   ├── manipulation.py  Arm-agnostic velocity-IK, batch IK, gripper control
│   └── coordinator.py   Generic outer task state machine
├── models/
│   └── combined.xml     Auto-generated MJCF (git-ignored if large)
├── scripts/
│   ├── build_model.py     MJCF generator
│   ├── run_simulation.py  Main entry point + task.type/task.sequence factory
│   └── smoke_test*.py     Quick sanity scripts
├── tasks/
│   ├── base.py           Task ABC: pluggable-task strategy interface
│   ├── reach_task.py     Task wrapper with logging
│   ├── pick_and_place.py PickAndPlaceTask: walk-grasp-transport-place
│   ├── reach_only.py     ReachOnlyTask: walk-and-reach, no grasp
│   └── push.py           PushTask: walk-and-push via real contact physics
├── tests/
│   ├── test_model.py
│   ├── test_controllers.py
│   ├── test_run_simulation.py
│   ├── test_stability.py
│   └── test_task.py
├── archieve/            Iterative development history (m01–m14)
│   └── controllers/     ik_controller_m0/m1/m2, grasp_controller, etc.
└── configs/
    └── default.yaml
```

---

## Technical Details

### Robot Model

- **Go2**: 12-DOF quadruped (4 legs × 3 joints). Free joint for base. Leg motors: hip/thigh ±60 Nm, knee ±90 Nm (boosted from stock for arm payload).
- **Arm (swappable, `--arm`)**: 7-DOF, rigidly mounted above Go2 base_link. Both options are registered in `controllers/arms.py` (`ArmSpec`) and selected via `scripts/build_model.py --arm <name>` / `scripts/run_simulation.py --arm <name>`:
  - **Franka Panda** (`franka`, default): masses scaled to 35% of original (~6.5 kg from 18.5 kg) to match real payload capacity. Integrated PD actuators (`general` type with `gainprm`/`biasprm`).
  - **Kinova Gen3** (`kinova_gen3`): native `position` PD servos (`kp`/`kv`), upstream masses/inertias preserved. Joints 1/3/5/7 are continuous (unranged) per the upstream model.
- **End-effector (swappable, `--end-effector`)**: registered in `controllers/end_effectors.py` (`EndEffectorSpec`):
  - **Franka hand** (`franka`): the Panda's stock two-finger gripper — Franka arm only.
  - **Robotiq 2F-85** (`robotiq_2f85`): vendored adaptive gripper, mountable on either arm via a per-arm `MountOverride` (Kinova's mount geometry differs from Franka's, and the Kinova mount omits the Robotiq base-mount body entirely).
  - **Kinova Gen3 only supports `robotiq_2f85`** — `validate_combo()` rejects `kinova_gen3` + `franka` with a `ValueError` before any model is built.
- **Combined model** (Franka + Franka-hand default): 35 qpos (7 base + 12 legs + 7 arm + 2 fingers + 7 cube), 20 actuators, ~21 kg total mass. Kinova + Robotiq: 41 qpos.

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

### Locomotion Turning

`WALKING` computes the bearing to the cube every step and steers the trot gait toward it via differential left/right stride amplitude (skid-steer style), instead of always walking straight in the body's local +X frame:

```python
bearing = atan2(cube_y - base_y, cube_x - base_x)
loco.set_heading(bearing)           # turns toward bearing every physics step
loco.set_speed_scale(scale)         # ramps stride amplitude down in the final approach
```

`base_yaw()` reads heading from the freejoint quaternion; `set_heading()` clips the proportional heading-error correction to `±1.0 rad/s` and applies it as a stride-amplitude scale on the right/left leg pairs (FR+RR vs. FL+RL). With zero heading error and full speed (the default demo's straight-ahead cube), this reduces to the original unmodified trot — no behavior change for the existing demo.

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
