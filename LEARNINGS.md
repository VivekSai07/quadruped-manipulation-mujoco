# Project Learnings — Quadruped Loco-Manipulation in MuJoCo

Everything learned while building an autonomous pick-and-place pipeline on a Unitree Go2 + Franka Panda system. Intended as a personal reference — detailed enough to reconstruct decisions from scratch.

---

## Table of Contents

1. [MuJoCo Fundamentals](#1-mujoco-fundamentals)
2. [MJCF XML Structure](#2-mjcf-xml-structure)
3. [Physics Options and Solver Settings](#3-physics-options-and-solver-settings)
4. [Go2 Quadruped Locomotion](#4-go2-quadruped-locomotion)
5. [Franka Panda Arm Control](#5-franka-panda-arm-control)
6. [Combining Two Robots in One Model](#6-combining-two-robots-in-one-model)
7. [Numerical Inverse Kinematics](#7-numerical-inverse-kinematics)
8. [Velocity IK — the m02 Pattern](#8-velocity-ik--the-m02-pattern)
9. [6-DOF Kinematic Attachment (Grasp Lock)](#9-6-dof-kinematic-attachment-grasp-lock)
10. [Task State Machine Design](#10-task-state-machine-design)
11. [Adaptive Height (Coordinated Loco-Manipulation)](#11-adaptive-height-coordinated-loco-manipulation)
12. [Contact Modeling and Grasp Physics](#12-contact-modeling-and-grasp-physics)
13. [Video Recording Pipeline](#13-video-recording-pipeline)
14. [Windows / conda Platform Quirks](#14-windows--conda-platform-quirks)
15. [Testing Strategy](#15-testing-strategy)
16. [Archive — Development Lineage (m01–m14)](#16-archive--development-lineage-m01m14)
17. [Key Numbers and Parameters](#17-key-numbers-and-parameters)

---

## 1. MuJoCo Fundamentals

### Data layout

MuJoCo separates model (static) from data (runtime state):

- `MjModel` — compiled MJCF: geometry, masses, joint limits, actuator gains. **Read-only at runtime.**
- `MjData` — simulation state: `qpos`, `qvel`, `ctrl`, `xpos`, `xmat`, contact forces, sensor readings. **Written every step.**

### Key arrays

| Array | What it holds |
|---|---|
| `data.qpos` | Generalized positions. Freejoints take 7 entries (x,y,z,qw,qx,qy,qz). Hinge joints take 1 entry each. |
| `data.qvel` | Generalized velocities. Freejoint: 6 entries (vx,vy,vz,wx,wy,wz). Hinge: 1 entry. |
| `data.ctrl` | Actuator control inputs. For `motor` type this is torque; for `general` type with `gainprm` it's desired position. |
| `data.xpos` | Body world-frame positions (indexed by body ID). |
| `data.xmat` | Body world-frame rotation matrices, 9-element row-major. |
| `data.site_xpos` | Site world-frame positions. Sites are point markers; preferred for EE tracking. |
| `data.site_xmat` | Site world-frame rotation matrices. |

### Step sequence

```python
mujoco.mj_step(model, data)   # one physics step: integrates, resolves contacts, updates xpos/xmat/etc.
```

Forward kinematics only (for IK without stepping):
```python
mujoco.mj_fwdPosition(model, data)   # updates xpos, xmat, site_xpos — no dynamics
```

### ID lookups

```python
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
act_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator1")
```

### qpos vs dof addresses

A joint's position in `qpos` and its velocity in `qvel` are at **different offsets**:
- `model.jnt_qposadr[jnt_id]` → index into `data.qpos`
- `model.jnt_dofadr[jnt_id]` → index into `data.qvel` (and Jacobian columns)

This distinction matters for freejoints (7 qpos entries, 6 dof entries) and for extracting arm Jacobian columns correctly.

---

## 2. MJCF XML Structure

MJCF (MuJoCo XML Format) defines the robot as a kinematic tree.

### Top-level elements

```xml
<mujoco model="name">
  <compiler/>       <!-- angle units, autolimits -->
  <option/>         <!-- timestep, solver settings -->
  <visual/>         <!-- rendering quality, offscreen size -->
  <default/>        <!-- default class hierarchy -->
  <asset/>          <!-- meshes, textures, materials -->
  <worldbody/>      <!-- the kinematic tree -->
  <actuator/>       <!-- motors / general actuators -->
  <tendon/>         <!-- coupled joints -->
  <equality/>       <!-- constraint equations -->
  <sensor/>         <!-- sensor definitions -->
  <contact/>        <!-- collision exclusions -->
  <keyframe/>       <!-- named initial states -->
</mujoco>
```

### Default class hierarchy

`<default class="go2">` sets inherited properties for all bodies tagged `childclass="go2"`. Child `<default>` blocks inside refine specific subsets:

```xml
<default class="go2">
  <motor ctrlrange="-60 60"/>
  <default class="knee">
    <motor ctrlrange="-90 90"/>   <!-- overrides parent for knee joints only -->
  </default>
</default>
```

This is cleaner than specifying every attribute on every geom/joint. Important gotcha: when merging two robots (Go2 + Panda), their default classes must have non-overlapping names or one silently overrides the other.

### Freejoints

A body with `<freejoint name="root"/>` has 6 DOF (3 translation + 3 rotation). Its qpos takes 7 entries (the quaternion). The robot base and any free-floating object (cube) both need freejoints.

### Sites vs geoms

- **Geom** — has collision shape and visual. Use for robot bodies and objects.
- **Site** — massless point marker (no collision). Use for end-effector position, IMU location, sensor attachment. Sites are the correct way to query EE position — they update with the kinematic chain without needing a forward pass.

### Keyframes

```xml
<keyframe>
  <key name="home" qpos="..." ctrl="..."/>
</keyframe>
```

Reset to a named keyframe with:
```python
keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
```

The `qpos` must list ALL joint positions in the exact order they appear in the tree (freejoint first, then each hinge in depth-first order).

---

## 3. Physics Options and Solver Settings

```xml
<option timestep="0.005" cone="elliptic" impratio="100"
        integrator="implicitfast" iterations="50" tolerance="1e-10"/>
```

### Timestep

`0.005 s` (200 Hz) is stable for this system. Lower is more accurate but slower. MuJoCo is unconditionally stable with implicit integration, so you can go higher (0.01 s) at the cost of contact accuracy.

### Integrator

- `Euler` — explicit, simple, can explode with stiff contacts.
- `implicitfast` — semi-implicit, stable for arm + leg dynamics at 5 ms. Recommended.
- `RK4` — 4th-order, 4× more function evaluations per step; rarely needed.

### Contact cone

- `pyramidal` — faster solver, less realistic friction (4/8 friction directions).
- `elliptic` — physically accurate friction cone. Use for manipulation (finger contacts).

### `impratio`

Contact impedance ratio. Higher values = stiffer contacts = less interpenetration, but harder for solver. `impratio="100"` significantly reduces cube sinking into the table and sliding out of the gripper.

### `solref` and `solimp` on geoms

Per-geom contact softness:
- `solref="0.002 1"` — contact time constant 2 ms (stiff), damping ratio 1.
- `solimp="0.9 0.95 0.001"` — near-rigid, small velocity-dependent softening.

Setting these on the cube geom prevents it from bouncing and sinking.

---

## 4. Go2 Quadruped Locomotion

### Joint structure (per leg)

Each leg has 3 joints (12 total for 4 legs):
- **Hip abduction** (`_hip_joint`): lateral sway, axis [1,0,0]
- **Hip flexion/extension** (`_thigh_joint`): forward/backward, axis [0,1,0]
- **Knee** (`_calf_joint`): flexion only (range constrained to negative values), axis [0,1,0]

Leg order in actuator array: FR, FL, RR, RL (front-right first, matching the original unitree_mujoco convention).

### PD control

Go2 uses torque actuators (`<motor>`). The controller computes torques externally:

```python
tau = kp * (q_des - q_meas) + kd * (0 - qd_meas)
data.ctrl[act_id] = tau
```

Gains: `kp_hip=60, kp_thigh=80, kp_knee=100, kd=4` (all in N·m/rad). These are per-joint-type, not per-leg. The knee uses higher kp because it bears most of the body weight.

### Standing pose

```python
_STAND_POSE = [0.006, 0.609, -1.218] * 4   # hip_ab, thigh, knee per leg
```

The thigh is ~35° from vertical, knee is ~70° (legs moderately bent). The `-` sign on knee is because the knee joint range is negative (bending constraint).

### Trot gait

Diagonal pairs move together: FR+RL swing simultaneously, FL+RR swing simultaneously. Phase offset = π between pairs. Sinusoidal joint targets:

```python
q_des[thigh_idx] = stand_pose[thigh] - ramp * 0.15 * sin(phi + phase_offset)
q_des[knee_idx]  = stand_pose[knee]  - ramp * 0.20 * sin(phi + phase_offset)
q_des[hip_idx]   = stand_pose[hip]   + ramp * 0.04 * cos(phi + phase_offset)
```

- Negative sine on thigh: swing (sin > 0) means foot reaches forward.
- Negative sine on knee: foot lifts during swing.
- Cosine on hip: lateral balance sway.

The `ramp = min(1, t_gait / 3.0)` prevents violent gait onset.

### Crouch blending

For adaptive height, we blend between two fixed poses:

```python
target = (1 - alpha) * _STAND_POSE + alpha * _CROUCH_POSE
```

`_CROUCH_POSE` has thigh=0.80 (was 0.609), knee=-1.50 (was -1.218). This lowers the body ~6 cm at alpha=1. We cap alpha at 0.5 in practice for safety.

### Height estimation

`data.qpos[2]` is the z-component of the freejoint position — the world-frame height of `base_link`. This is valid immediately after `mj_resetDataKeyframe` without needing a forward pass.

---

## 5. Franka Panda Arm Control

### Actuator type: `general` with `gainprm`/`biasprm`

Unlike Go2 (which uses torque motors and computes PD externally), the Panda uses MuJoCo's built-in PD:

```xml
<general name="actuator1" joint="joint1"
         gainprm="4500" biasprm="0 -4500 -450"/>
```

- `gainprm="4500"` → force = 4500 × ctrl
- `biasprm="0 -4500 -450"` → bias = 0 - 4500 × q - 450 × qd (spring + damper)
- Net: force = 4500 × (ctrl - q) - 450 × qd → this is kp=4500, kd=450 position control

**The ctrl input is the desired joint angle in radians.** MuJoCo computes the torque internally. This means the controller only needs to set `data.ctrl[act_id] = q_desired`.

### Gripper actuator

The gripper uses a tendon to synchronize two finger joints:

```xml
<tendon>
  <fixed name="panda_split">
    <joint joint="finger_joint1" coef="0.5"/>
    <joint joint="finger_joint2" coef="0.5"/>
  </fixed>
</tendon>
<general name="actuator8" tendon="panda_split"
         gainprm="0.01568..." ctrlrange="0 255"/>
```

`ctrl=255` → fingers fully open (0.04 m gap). `ctrl=0` → fingers closed (spring pulls to 0). The 0.01568... is 4/255 so ctrl=255 maps to 0.04 m target position.

### Home pose

```python
_HOME_POSE = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853]
```

This folds the arm safely above the robot body (joint4=-90°, joint6=90°, joint7=-45°). It's the default rest position and the target for RETURNING_HOME.

### Joint limits

Important: `joint4` (elbow pitch) has range `[-3.0718, -0.0698]` — **always negative**. This is unusual and must be respected in IK clipping. Violating it causes the elbow to flip.

### Mass scaling

The real Panda masses ~18.5 kg. The Go2 payload limit is ~5–8 kg. We scaled all Panda inertias by 0.35 at model-build time, reducing the simulated arm to ~6.5 kg. This prevents the robot from toppling during manipulation.

```python
PANDA_MASS_SCALE = 0.35
# Applied to all <inertial mass="..."> and fullinertia="..." values in Panda XML
```

### Motor limits boost

The Go2's stock hip/thigh motor limit was ±40 Nm, insufficient to stand stably with the heavier arm. Boosted to ±60 Nm (hip/thigh) and ±90 Nm (knee) in the `<default class="go2">` block.

---

## 6. Combining Two Robots in One Model

### Strategy: generate XML programmatically

Rather than merging two hand-crafted XML files (fragile, hard to update), `build_model.py` generates the entire `combined.xml` as a Python string. Benefits:
- Mass scaling applied at generation time (simple string arithmetic on inertia values)
- No merge conflicts between Go2 and Panda XML namespaces
- Reproducible: run `python scripts/build_model.py` to regenerate from scratch

### Mounting the arm

```xml
<body name="panda_link0" pos="0 0 0.10" childclass="panda">
```

Inside `base_link` (Go2 trunk). No joint → rigid attachment. `pos="0 0 0.10"` places the arm base 10 cm above the trunk center (above the trunk top face at +0.057 m). `childclass="panda"` overrides the inherited `go2` defaults for all arm bodies.

### qpos layout (combined model)

```
[0:3]   base_link xyz        (freejoint translation)
[3:7]   base_link quat       (freejoint rotation)
[7:19]  leg joints × 12      (FR/FL/RR/RL × hip/thigh/knee)
[19:26] panda joint1–7
[26:28] finger_joint1, finger_joint2
[28:35] cube freejoint (xyz + quat)
```

The cube freejoint must be declared **last** in worldbody so `qpos[28:35]` addresses stay predictable. The base_link freejoint is always first at `qpos[0:7]`.

### ctrl layout

```
[0:12]   Go2 leg torques (FR/FL/RR/RL × hip/thigh/calf)
[12:19]  Panda arm desired angles (joint1–7)
[19]     Gripper (0=closed, 255=open)
```

### Namespace conflicts

Go2 XML and Panda XML both define material names like `"white"`, `"metal"`. Solution: prefix all Panda materials with `p_` (`p_white`, `p_dark`, etc.). Same for mesh names (`p_link0_0` etc.).

### Contact exclusions

Adjacent Panda links must be excluded from self-collision (they overlap by design):
```xml
<contact>
  <exclude body1="panda_link0" body2="panda_link1"/>
  ...
  <exclude body1="base_link"   body2="panda_link0"/>  <!-- mounting interface -->
</contact>
```

---

## 7. Numerical Inverse Kinematics

### Jacobian

The IK Jacobian relates joint velocities to end-effector velocities:

```
dx/dt = J(q) * dq/dt
```

`mujoco.mj_jacSite` computes the full 6×nv Jacobian (3 position rows + 3 rotation rows). We extract only the 7 arm DOF columns:

```python
J_full = np.zeros((6, model.nv))
mujoco.mj_jacSite(model, data, J_full[:3], J_full[3:], ee_site_id)
J_arm = J_full[:n_task, col_ids]   # n_task=6 for 6DOF, col_ids = arm dof addresses
```

### Levenberg-Marquardt (batch IK)

Solve `dq = J^T (J J^T + λ²I)^{-1} err` at each IK iteration:

```python
JJT = J_arm @ J_arm.T
dq  = J_arm.T @ np.linalg.solve(JJT + lambda_sq * I, error)
q   = np.clip(q + step_size * dq, Q_LO, Q_HI)
```

The damping term `λ²I` prevents joint velocity blowup near singularities. `λ=0.01` works well for Panda.

### 6-DOF vs 3-DOF

For manipulation we use 6-DOF IK (position + orientation). The `error` vector is:
```python
error = np.concatenate([pos_err_3d, rot_gain * rot_err_3d])
```

`rot_err_3d` is the axis-angle vector from current to target orientation, computed via:
```python
R_err = R_target @ R_current.T
theta = arccos((trace(R_err) - 1) / 2)
rot_err = (theta / (2 * sin(theta))) * [R_err[2,1]-R_err[1,2], ...]
```

Use `rot_gain < 1` (we use 0.4) so position convergence is not sacrificed for orientation.

### Target orientation

For top-down grasping, the EE should point straight down:
```python
_TARGET_ROTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
])
```

Column 3 (local-Z / approach direction) = world [0,0,-1] = pointing down.

### Why batch IK causes jerky motion

Batch IK is called infrequently (e.g., every 0.3 s) and computes a target jump of several cm. The arm PD controller then races to the new target in one burst → visible 3 cm lurch. The fix is velocity IK.

---

## 8. Velocity IK — the m02 Pattern

The key insight from archive mark m02: instead of computing a complete IK solution every N seconds, integrate a single Jacobian step every physics timestep.

### Algorithm (one step per dt=0.005s)

```python
# 1. FK at measured qpos (not commanded target) for accurate Jacobian
q_meas = arm_qpos()
for i, adr in enumerate(jnt_qadr):
    data.qpos[adr] = q_meas[i]
mujoco.mj_fwdPosition(model, data)

# 2. Compute error
pos_err = target - site_xpos[ee_site_id]

# 3. Desired task-space velocity
v_des = [Kp * pos_err, Kr * rot_gain * rot_err]   # 6D

# 4. Jacobian pseudo-inverse
dq = J.T @ solve(J @ J.T + lambda^2 * I, v_des)
dq = clip(dq, -max_vel, max_vel)

# 5. Integrate from COMMANDED target (not measured qpos!)
q_target += dq * dt

# 6. Restore physics state
data.qpos[:] = saved_qpos
```

### Why integrate from commanded target

If you integrate from `arm_qpos()` (measured), the step is only `~0.0075 rad` ahead of the arm that's lagging behind at `~0.004 rad/step`. The target barely moves. If you integrate from `_q_target` (commanded), the target races ahead of the arm at full velocity regardless of PD lag. This is the critical fix for RETURNING_HOME and for fast smooth approach.

### State save/restore

IK temporarily modifies `data.qpos`, `data.qvel`, `data.ctrl` to run forward kinematics. Always save and restore:
```python
qpos_save, qvel_save, ctrl_save = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
# ... run mj_fwdPosition ...
data.qpos[:], data.qvel[:], data.ctrl[:] = qpos_save, qvel_save, ctrl_save
mujoco.mj_fwdPosition(model, data)   # restore consistent FK state
```

Forgetting this corrupts the physics state and causes the robot to teleport.

### Cartesian interp target

On top of velocity IK, we maintain an intermediate Cartesian target that moves at a fixed rate (10 cm/s):

```python
step  = ARM_MOVE_RATE * dt          # 0.10 * 0.005 = 0.0005 m/step
delta = goal - interp_target
dist  = norm(delta)
if dist <= step:
    interp_target = goal
else:
    interp_target += delta / dist * step
```

The velocity IK tracks this moving point. This gives smooth continuous motion because the tracked target never jumps.

---

## 9. 6-DOF Kinematic Attachment (Grasp Lock)

### Problem

When the gripper closes on the cube, physics contact forces are still computed. During transport, the cube can slide or tumble if contact forces are insufficient (high transport speed, imperfect gripper geometry).

### Solution: override physics with kinematic constraint

At grasp time, record the cube's pose relative to the EE:

```python
ee_xmat   = site_xmat[ee_site_id].reshape(3,3)
cube_xmat = data.xmat[cube_body_id].reshape(3,3)
cube_pos  = data.qpos[cube_qpos_adr:cube_qpos_adr+3]

_grasp_offset  = cube_pos - ee_position()    # translation offset in world frame
_grasp_R_local = ee_xmat.T @ cube_xmat      # cube rotation in EE local frame
```

Every physics step during LIFTING/TRANSPORTING/LOWERING, snap cube to EE:

```python
ee_pos  = ee_position()
ee_xmat = site_xmat[ee_site_id].reshape(3,3)

new_cube_pos  = ee_pos + _grasp_offset
new_cube_R    = ee_xmat @ _grasp_R_local          # reconstruct world rotation
new_cube_quat = mat2quat(new_cube_R)              # Shepperd's method

data.qpos[a:a+3] = new_cube_pos
data.qpos[a+3:a+7] = new_cube_quat
data.qvel[v:v+6]  = 0.0    # zero velocity so solver can't fight the constraint
```

Do this **twice per step**: once before `mj_step` (in the controller) and once after (in `post_physics_step`) because the constraint solver can nudge the cube during the contact resolution pass.

### Shepperd's method (mat → quat)

MuJoCo stores quaternions as [w, x, y, z]. Standard Shepperd's method:

```python
def mat2quat(R):
    t = trace(R)
    if t > 0:
        s = 0.5 / sqrt(t + 1)
        return [0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s]
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2 * sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        return [(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]
    ...  # similarly for other dominant diagonal element cases
```

Use the case with the largest diagonal element to avoid numerical instability near singularities.

---

## 10. Task State Machine Design

### 15 states

```
INIT → STANDING → WALKING → STOPPING → STABILIZING
     → ADJUSTING_HEIGHT → APPROACHING → DESCENDING → GRASPING
     → LIFTING → TRANSPORTING → LOWERING → RELEASING → RETURNING_HOME → DONE
```

### Why so many states

Each state transition requires different stopping conditions and different arm behavior. Merging states (e.g., combining APPROACHING and DESCENDING) makes timing logic complicated and harder to tune individually.

### Minimum time guards

Every motion state has a minimum time before its termination condition is checked:

```python
if elapsed >= min_approach_time and dist < approach_threshold:
    transition(DESCENDING)
```

Without `min_approach_time`, the arm might register as "at waypoint" before it has left the previous waypoint (because velocity IK has not moved it yet at t=0 of the state).

### Seeding the interp target

On every state entry that starts arm motion, seed `_arm_interp_target` at the current EE position:

```python
def _seed_approach(self, t):
    self._arm_interp_target = self.manip.ee_position().copy()
```

This ensures the first velocity-IK step is a zero-length move. Without seeding, the interp target might be at the previous state's goal, causing the arm to snap backward on state entry.

### Status line

Print a single-line status every 0.5 s with all relevant metrics:

```
t= 9.07s | adjusting_height  | h=0.431m | base->cube=0.647m | ee->approach=0.387m | cube_z=0.325m | grasped=False
```

This is invaluable for diagnosing which state is running long and what the blocking metric is.

---

## 11. Adaptive Height (Coordinated Loco-Manipulation)

### Motivation

The Go2's base is ~0.43 m above the ground. The cube is on a table at 0.30 m (table surface) + 0.025 m (cube half-height) = 0.325 m. The base is already above the cube. But the arm must reach forward ~65 cm to the table, increasing torque on the Go2 body. Lowering slightly improves balance and expands the arm's useful workspace downward.

### Key insight: compute, don't hardcode

The crouch amount depends on the actual reached distance after walking, not a fixed value. Different scenarios (cube closer/farther, table taller/shorter) should automatically produce different crouch amounts.

```python
horizontal_reach = abs(wp_descend_x - robot_x)
vertical_reach   = max(0, base_z - wp_descend_z)   # how far EE must go below base

alpha_h = max(0, (horizontal_reach - 0.30) / 0.60)  # 0 at 30cm, 1 at 90cm reach
alpha_v = max(0, (vertical_reach   - 0.00) / 0.30)  # 0 at base level, 1 at 30cm below

alpha = min(0.5, alpha_h * 0.35 + alpha_v * 0.25)   # weighted sum, capped at 0.5
```

For the default scenario: horizontal=0.64 m → alpha_h=0.567, vertical=0.05 m → alpha_v=0.167.
Final: min(0.5, 0.567×0.35 + 0.167×0.25) = min(0.5, 0.240) = **0.24**. About 2–3 cm lower.

### Allow settle time

After calling `loco.set_crouch_alpha(alpha)`, wait 2 s (`height_settle_time`) before starting arm motion. The PD controller needs several hundred milliseconds to move the legs to the new pose and damp out oscillations.

---

## 12. Contact Modeling and Grasp Physics

### Contact detection

MuJoCo stores all active contacts in `data.contact[0:data.ncon]`. Each contact has `.geom1` and `.geom2` (geom IDs). Body ID from geom: `model.geom_bodyid[geom_id]`.

```python
def is_grasped():
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if {b1,b2} == {left_finger_id, cube_id}:
            left_touch = True
        if {b1,b2} == {right_finger_id, cube_id}:
            right_touch = True
    return left_touch and right_touch
```

Both fingers must contact the cube, not just one.

### Fingertip contact geoms

The Franka hand has a flat fingertip surface. We added 5 small box geoms per finger at the pad and tip positions with high friction:

```xml
<geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445"
      friction="1.5 0.05 0.01" condim="6" class="ftp1"/>
```

`condim="6"` enables full 3D friction (including torsional and rolling resistance). This is critical for gripping without the cube spinning.

### Cube physics properties

```xml
<geom type="box" size="0.025 0.025 0.025"
      friction="1.5 0.05 0.01" condim="6"
      solref="0.002 1" solimp="0.9 0.95 0.001"/>
```

- High friction (1.5) prevents sliding on the table and in the gripper.
- Stiff contact (`solref`, `solimp`) prevents the cube from sinking into surfaces.
- Low mass (0.1 kg, `I=4.17e-5`) means finger forces easily dominate.

### Why kinematic attachment is still needed

Even with good friction settings, the gripper's contact area is small (mm-scale geoms), and rapid arm acceleration during LIFTING can momentarily overcome friction forces. The kinematic attachment overrides physics entirely, guaranteeing zero slip regardless of acceleration.

---

## 13. Video Recording Pipeline

### Components

1. `mujoco.Renderer(model, height=H, width=W)` — offscreen OpenGL renderer
2. `renderer.update_scene(data, camera=cam)` — renders current state from camera view
3. `renderer.render()` — returns `(H, W, 3)` uint8 numpy array (RGB)
4. `cv2.VideoWriter` — encodes frames to MP4

### Camera setup

```python
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type     = mujoco.mjtCamera.mjCAMERA_FREE
cam.azimuth  = -140.0
cam.elevation = -20.0
cam.distance = 3.5
cam.lookat[:] = [0.5, 0.0, 0.3]
```

`mjCAMERA_FREE` is a free-floating camera (not attached to a body). Azimuth/elevation/distance define the spherical coordinate view.

### Offscreen framebuffer size

MuJoCo's Renderer uses an OpenGL framebuffer sized by `<global offwidth offheight>` in the model XML:

```xml
<global azimuth="160" elevation="-20" offwidth="1280" offheight="720"/>
```

If you request `Renderer(model, height=720, width=1280)` but the model has `offwidth=640`, MuJoCo raises `ValueError: Image width 1280 > framebuffer width 640`. The fix is in `build_model.py`, not in the renderer call.

### Frame rate

To record at 30 fps from a 200 Hz simulation:
```python
record_every = round(1.0 / (fps * dt))   # = round(1/(30 * 0.005)) = round(6.67) = 7
```

Record one frame every 7 steps → actual fps = 1/(7 * 0.005) = 28.6 fps (close enough).

### OpenCV BGR

OpenCV uses BGR, MuJoCo renders RGB. Convert before writing:
```python
rgb = renderer.render()       # (H, W, 3) RGB
bgr = rgb[:, :, ::-1]         # flip channels: RGB → BGR
out.write(bgr)
```

### Video duration

The simulation stops when `task.is_done` — the video automatically ends at task completion regardless of `--duration`. Never pass `--duration` shorter than the task takes, or the recording will be cut off.

---

## 14. Windows / conda Platform Quirks

### stdout suppression via `conda run`

`conda run -n base python script.py` on Windows swallows stdout (Python's default line-buffered mode buffers output until the buffer fills). Solution at the top of the script:

```python
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
```

This forces every `print()` call to flush immediately. Only stderr (Python warnings) passes through without this fix.

### Unicode em-dashes on Windows

Windows terminal and Python's cp1252 encoding cannot display the Unicode em-dash `—` (U+2014). Use ASCII double-hyphen `--` instead in all print statements. The error appears as garbled characters or `UnicodeEncodeError`.

### conda environment path issues

When running via `conda run`, the sys.path may not include the project root. Fix with:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

at the top of any script that imports from the controllers package.

### PowerShell vs bash

On Windows the project runs under PowerShell. Environment variable syntax is `$env:VAR`, not `$VAR`. Pipeline chain `&&` is not available in PowerShell 5.1 — use `; if ($?) { ... }` instead.

---

## 15. Testing Strategy

### Test files

| File | What it covers |
|---|---|
| `test_model.py` | Model loads, nq/nu/nbody counts, keyframe reset, qpos layout |
| `test_controllers.py` | LocomotionController PD output, ManipulationController IK, `is_grasped()` |
| `test_stability.py` | Robot stands stably (height > threshold) after N steps |
| `test_task.py` | Full task integration: does coordinator reach DONE within timeout? |

### Running tests

```bash
pytest tests/ -v                   # all 39 tests
pytest tests/test_model.py -v      # model only
pytest tests/ -k "stability" -v    # filter by name
```

### What not to test

- Exact pixel values of the rendered image (too brittle, GPU-dependent)
- Exact simulation trajectory (non-deterministic contact resolution can vary)
- Timing down to the millisecond (depends on timestep + platform)

Instead test **structural guarantees**: does the state machine reach DONE? Does IK converge within N iterations? Is the model's nq correct?

---

## 16. Archive — Development Lineage (m01–m14)

The `archieve/` directory contains the full Franka Panda standalone development history before Go2 was added.

| Mark | Key contribution |
|---|---|
| m01 | Basic position IK, first working reach |
| m02 | Velocity (nullspace) IK — smooth continuous motion. This pattern was carried forward into the final system. |
| m03 | Trajectory planner with waypoints |
| m04 | Sense-plan-recover loop (detect failure, replan) |
| m05 | RRT path planning around obstacles |
| m06 | Keyboard teleoperation (WASD + arm control) |
| m07 | Machine vision — camera, point cloud, object detection |
| m08 | Autonomous obstacle avoidance using APF (Artificial Potential Fields) |
| m09 | YOLO-based object detection for pick target |
| m10 | MoCap teleoperation (record + replay human demonstrations) |
| m11 | VLM-guided pick-and-place (natural language → action) |
| m12 | Florence-2 vision-language model integration |
| m13 | Reactive manipulation (online replanning from sensor feedback) |
| m14 | TAMP (Task and Motion Planning) — symbolic + geometric planning |

The m02 velocity IK pattern was specifically referenced when fixing the jerky EE motion in the final system.

---

## 17. Key Numbers and Parameters

### Robot dimensions

| Quantity | Value |
|---|---|
| Go2 base height (standing) | ~0.43 m |
| Go2 base height (crouched, alpha=1) | ~0.37 m |
| Panda link0 mount offset from base_link | 0.10 m |
| Panda reach (fully extended) | ~0.85 m |
| Table height (top surface) | 0.30 m |
| Cube center height | 0.325 m (table + half-size) |
| Placement plate center height | 0.331 m |

### Physics timestep and rates

| Quantity | Value |
|---|---|
| Physics timestep (dt) | 0.005 s (200 Hz) |
| Arm Cartesian move rate | 0.10 m/s (10 cm/s) |
| Arm vertical lift/lower rate | 0.04 m/s (4 cm/s) |
| Joint return rate (RETURNING_HOME) | 1.5 rad/s |
| Gait period | 0.6 s |
| Gait ramp time | 3.0 s |

### IK parameters

| Parameter | Value | Notes |
|---|---|---|
| `_VEL_IK_KP` | 5.0 | Translational proportional gain |
| `_VEL_IK_KR` | 3.0 | Rotational proportional gain |
| `_VEL_IK_MAX` | 0.8 rad/s | Joint velocity clip |
| `_IK_ROT_GAIN` | 0.4 | Rotation error weight vs. position |
| `IK_DAMPING` | 0.01 | Levenberg-Marquardt λ |
| `IK_TOLERANCE` | 0.003 m | 3 mm convergence threshold |

### Task timing (observed)

| State | Duration |
|---|---|
| STANDING | ~2 s (ramp + stable check) |
| WALKING | ~5 s (0.65 m at trot speed) |
| STOPPING + STABILIZING | ~4 s |
| ADJUSTING_HEIGHT | 2 s settle |
| APPROACHING | ~5–6 s (48 cm at 10 cm/s) |
| DESCENDING | ~3–4 s (15 cm at 10 cm/s) |
| GRASPING | 3 s hold |
| LIFTING | ~4 s (15 cm at 4 cm/s) |
| TRANSPORTING | ~3 s (20 cm lateral) |
| LOWERING | ~4 s |
| RELEASING | 0.8 s |
| RETURNING_HOME | ~1.7 s (1.1 rad total at 1.5 rad/s) |
| **Total** | **~34 s** |
