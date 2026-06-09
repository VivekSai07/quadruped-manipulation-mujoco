"""
Build the combined Go2 + Franka Panda MJCF model.

Reads source XML files from assets/, applies:
 - Mass scaling on Panda (×0.35) to match Go2 payload capacity
 - Boosted Go2 motor limits for heavier payload
 - Merged default classes with unambiguous names
 - Explicit mesh paths relative to models/ directory
 - Target cube and end-effector site
Writes models/combined.xml.
"""
from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Panda mass scale factor: reduces ~18.5 kg arm to ~6.5 kg
PANDA_MASS_SCALE = 0.35


def _scale_inertia(value_str: str) -> str:
    """Scale a space-separated inertia string by PANDA_MASS_SCALE."""
    parts = value_str.strip().split()
    scaled = [f"{float(p) * PANDA_MASS_SCALE:.6g}" for p in parts]
    return " ".join(scaled)


def build_combined_xml() -> str:
    """Return the complete combined MJCF XML as a string."""

    xml = """<mujoco model="go2_panda">
  <!--
    Combined Unitree Go2 + Franka Panda loco-manipulation model.
    Coordinate frame: X-forward, Y-left, Z-up (standard robotics).
    Panda link0 is rigidly mounted at pos="0 0 0.10" on Go2 base_link.
    Panda masses scaled by 0.35 (original ~18.5 kg → ~6.5 kg).
    Go2 motor limits boosted: hip/thigh ±60 Nm, knee ±90 Nm.
  -->

  <!-- ── Compiler ─────────────────────────────────────────────────────── -->
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>

  <!-- ── Physics options ─────────────────────────────────────────────── -->
  <option timestep="0.005" cone="elliptic" impratio="100"
          integrator="implicitfast" iterations="50" tolerance="1e-10"/>

  <!-- ── Visual quality ──────────────────────────────────────────────── -->
  <visual>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <map shadowclip="2.0" shadowscale="0.6" fogstart="8" fogend="20"/>
    <rgba haze="0.18 0.22 0.30 1"/>
    <global azimuth="160" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <!-- ── Default classes ─────────────────────────────────────────────── -->
  <default>

    <!-- Go2 robot defaults -->
    <default class="go2">
      <geom friction="0.4" margin="0.001" condim="1"/>
      <joint axis="0 1 0" damping="0.1" armature="0.01" frictionloss="0.2"/>
      <motor ctrlrange="-60 60"/>
      <default class="abduction">
        <joint axis="1 0 0" range="-1.0472 1.0472"/>
      </default>
      <default class="front_hip">
        <joint range="-1.5708 3.4907"/>
      </default>
      <default class="back_hip">
        <joint range="-0.5236 4.5379"/>
      </default>
      <default class="knee">
        <joint range="-2.7227 -0.83776"/>
        <motor ctrlrange="-90 90"/>
      </default>
      <default class="go2_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="go2_collision">
        <geom group="3"/>
        <default class="foot">
          <geom size="0.022" pos="-0.002 0 -0.213" priority="1" condim="6"
            friction="0.4 0.02 0.01"/>
        </default>
      </default>
    </default>

    <!-- Franka Panda defaults -->
    <default class="panda">
      <material specular="0.5" shininess="0.25"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973"
               forcerange="-87 87"/>
      <default class="panda_finger">
        <joint axis="0 1 0" type="slide" range="0 0.04"/>
      </default>
      <default class="panda_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="panda_collision">
        <geom type="mesh" group="3"/>
        <default class="ftp1">
          <geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445"
                friction="1.5 0.05 0.01" condim="6"/>
        </default>
        <default class="ftp2">
          <geom type="box" size="0.003 0.002 0.003" pos="0.0055 0.002 0.05"
                friction="1.5 0.05 0.01" condim="6"/>
        </default>
        <default class="ftp3">
          <geom type="box" size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05"
                friction="1.5 0.05 0.01" condim="6"/>
        </default>
        <default class="ftp4">
          <geom type="box" size="0.003 0.002 0.0035" pos="0.0055 0.002 0.0395"
                friction="1.5 0.05 0.01" condim="6"/>
        </default>
        <default class="ftp5">
          <geom type="box" size="0.003 0.002 0.0035" pos="-0.0055 0.002 0.0395"
                friction="1.5 0.05 0.01" condim="6"/>
        </default>
      </default>
    </default>

  </default>

  <!-- ── Assets ──────────────────────────────────────────────────────── -->
  <asset>
    <!-- Skybox: cool blue-to-dark gradient for a studio feel -->
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1=".45 .60 .80" rgb2=".08 .10 .18" width="512" height="512"/>

    <!-- Floor: grey checker tile -->
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1=".82 .82 .82" rgb2=".65 .65 .65"
             width="512" height="512" mark="cross" markrgb=".75 .75 .75"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 6"
              reflectance="0.08" specular="0.1" shininess="0.1"/>

    <!-- Table: warm oak-like wood colour -->
    <material name="table_mat"  rgba="0.72 0.52 0.32 1"
              specular="0.15" shininess="0.25" reflectance="0.05"/>
    <material name="table_leg_mat" rgba="0.52 0.36 0.18 1" specular="0.1"/>

    <!-- Cube: vivid red so it stands out -->
    <material name="cube_mat"   rgba="0.90 0.15 0.12 1"
              specular="0.4" shininess="0.5" reflectance="0.15"/>

    <!-- Placement plate: bright green target zone -->
    <material name="plate_mat"  rgba="0.15 0.80 0.25 1"
              specular="0.3" shininess="0.4" reflectance="0.1"/>

    <!-- Materials: Go2 -->
    <material name="metal"  rgba=".82 .86 .88 1" specular="0.5" shininess="0.3"/>
    <material name="black"  rgba="0.10 0.10 0.10 1" specular="0.3" shininess="0.2"/>
    <material name="white"  rgba="0.92 0.92 0.92 1" specular="0.2" shininess="0.15"/>
    <material name="gray"   rgba="0.50 0.52 0.58 1" specular="0.25" shininess="0.2"/>

    <!-- Materials: Panda (prefixed to avoid conflict with Go2 names) -->
    <material name="p_white"      rgba="0.94 0.94 0.94 1" specular="0.3" shininess="0.2"/>
    <material name="p_off_white"  rgba="0.88 0.90 0.91 1" specular="0.25" shininess="0.15"/>
    <material name="p_dark"       rgba="0.20 0.20 0.22 1" specular="0.4" shininess="0.35"/>
    <material name="p_green"      rgba="0 0.80 0.10 1" specular="0.3"/>
    <material name="p_light_blue" rgba="0.04 0.54 0.78 1" specular="0.4" shininess="0.4"/>

    <!-- Go2 meshes (relative to models/ dir) -->
    <mesh name="base_0"         file="../assets/go2/assets/base_0.obj"/>
    <mesh name="base_1"         file="../assets/go2/assets/base_1.obj"/>
    <mesh name="base_2"         file="../assets/go2/assets/base_2.obj"/>
    <mesh name="base_3"         file="../assets/go2/assets/base_3.obj"/>
    <mesh name="base_4"         file="../assets/go2/assets/base_4.obj"/>
    <mesh name="hip_0"          file="../assets/go2/assets/hip_0.obj"/>
    <mesh name="hip_1"          file="../assets/go2/assets/hip_1.obj"/>
    <mesh name="thigh_0"        file="../assets/go2/assets/thigh_0.obj"/>
    <mesh name="thigh_1"        file="../assets/go2/assets/thigh_1.obj"/>
    <mesh name="thigh_mirror_0" file="../assets/go2/assets/thigh_mirror_0.obj"/>
    <mesh name="thigh_mirror_1" file="../assets/go2/assets/thigh_mirror_1.obj"/>
    <mesh name="calf_0"         file="../assets/go2/assets/calf_0.obj"/>
    <mesh name="calf_1"         file="../assets/go2/assets/calf_1.obj"/>
    <mesh name="calf_mirror_0"  file="../assets/go2/assets/calf_mirror_0.obj"/>
    <mesh name="calf_mirror_1"  file="../assets/go2/assets/calf_mirror_1.obj"/>
    <mesh name="go2_foot"       file="../assets/go2/assets/foot.obj"/>

    <!-- Panda collision meshes (named with p_ prefix) -->
    <mesh name="p_link0_c"  file="../assets/panda/assets/link0.stl"/>
    <mesh name="p_link1_c"  file="../assets/panda/assets/link1.stl"/>
    <mesh name="p_link2_c"  file="../assets/panda/assets/link2.stl"/>
    <mesh name="p_link3_c"  file="../assets/panda/assets/link3.stl"/>
    <mesh name="p_link4_c"  file="../assets/panda/assets/link4.stl"/>
    <mesh name="p_link5_c0" file="../assets/panda/assets/link5_collision_0.obj"/>
    <mesh name="p_link5_c1" file="../assets/panda/assets/link5_collision_1.obj"/>
    <mesh name="p_link5_c2" file="../assets/panda/assets/link5_collision_2.obj"/>
    <mesh name="p_link6_c"  file="../assets/panda/assets/link6.stl"/>
    <mesh name="p_link7_c"  file="../assets/panda/assets/link7.stl"/>
    <mesh name="p_hand_c"   file="../assets/panda/assets/hand.stl"/>

    <!-- Panda visual meshes -->
    <mesh name="p_link0_0"  file="../assets/panda/assets/link0_0.obj"/>
    <mesh name="p_link0_1"  file="../assets/panda/assets/link0_1.obj"/>
    <mesh name="p_link0_2"  file="../assets/panda/assets/link0_2.obj"/>
    <mesh name="p_link0_3"  file="../assets/panda/assets/link0_3.obj"/>
    <mesh name="p_link0_4"  file="../assets/panda/assets/link0_4.obj"/>
    <mesh name="p_link0_5"  file="../assets/panda/assets/link0_5.obj"/>
    <mesh name="p_link0_7"  file="../assets/panda/assets/link0_7.obj"/>
    <mesh name="p_link0_8"  file="../assets/panda/assets/link0_8.obj"/>
    <mesh name="p_link0_9"  file="../assets/panda/assets/link0_9.obj"/>
    <mesh name="p_link0_10" file="../assets/panda/assets/link0_10.obj"/>
    <mesh name="p_link0_11" file="../assets/panda/assets/link0_11.obj"/>
    <mesh name="p_link1"    file="../assets/panda/assets/link1.obj"/>
    <mesh name="p_link2"    file="../assets/panda/assets/link2.obj"/>
    <mesh name="p_link3_0"  file="../assets/panda/assets/link3_0.obj"/>
    <mesh name="p_link3_1"  file="../assets/panda/assets/link3_1.obj"/>
    <mesh name="p_link3_2"  file="../assets/panda/assets/link3_2.obj"/>
    <mesh name="p_link3_3"  file="../assets/panda/assets/link3_3.obj"/>
    <mesh name="p_link4_0"  file="../assets/panda/assets/link4_0.obj"/>
    <mesh name="p_link4_1"  file="../assets/panda/assets/link4_1.obj"/>
    <mesh name="p_link4_2"  file="../assets/panda/assets/link4_2.obj"/>
    <mesh name="p_link4_3"  file="../assets/panda/assets/link4_3.obj"/>
    <mesh name="p_link5_0"  file="../assets/panda/assets/link5_0.obj"/>
    <mesh name="p_link5_1"  file="../assets/panda/assets/link5_1.obj"/>
    <mesh name="p_link5_2"  file="../assets/panda/assets/link5_2.obj"/>
    <mesh name="p_link6_0"  file="../assets/panda/assets/link6_0.obj"/>
    <mesh name="p_link6_1"  file="../assets/panda/assets/link6_1.obj"/>
    <mesh name="p_link6_2"  file="../assets/panda/assets/link6_2.obj"/>
    <mesh name="p_link6_3"  file="../assets/panda/assets/link6_3.obj"/>
    <mesh name="p_link6_4"  file="../assets/panda/assets/link6_4.obj"/>
    <mesh name="p_link6_5"  file="../assets/panda/assets/link6_5.obj"/>
    <mesh name="p_link6_6"  file="../assets/panda/assets/link6_6.obj"/>
    <mesh name="p_link6_7"  file="../assets/panda/assets/link6_7.obj"/>
    <mesh name="p_link6_8"  file="../assets/panda/assets/link6_8.obj"/>
    <mesh name="p_link6_9"  file="../assets/panda/assets/link6_9.obj"/>
    <mesh name="p_link6_10" file="../assets/panda/assets/link6_10.obj"/>
    <mesh name="p_link6_11" file="../assets/panda/assets/link6_11.obj"/>
    <mesh name="p_link6_12" file="../assets/panda/assets/link6_12.obj"/>
    <mesh name="p_link6_13" file="../assets/panda/assets/link6_13.obj"/>
    <mesh name="p_link6_14" file="../assets/panda/assets/link6_14.obj"/>
    <mesh name="p_link6_15" file="../assets/panda/assets/link6_15.obj"/>
    <mesh name="p_link6_16" file="../assets/panda/assets/link6_16.obj"/>
    <mesh name="p_link7_0"  file="../assets/panda/assets/link7_0.obj"/>
    <mesh name="p_link7_1"  file="../assets/panda/assets/link7_1.obj"/>
    <mesh name="p_link7_2"  file="../assets/panda/assets/link7_2.obj"/>
    <mesh name="p_link7_3"  file="../assets/panda/assets/link7_3.obj"/>
    <mesh name="p_link7_4"  file="../assets/panda/assets/link7_4.obj"/>
    <mesh name="p_link7_5"  file="../assets/panda/assets/link7_5.obj"/>
    <mesh name="p_link7_6"  file="../assets/panda/assets/link7_6.obj"/>
    <mesh name="p_link7_7"  file="../assets/panda/assets/link7_7.obj"/>
    <mesh name="p_hand_0"   file="../assets/panda/assets/hand_0.obj"/>
    <mesh name="p_hand_1"   file="../assets/panda/assets/hand_1.obj"/>
    <mesh name="p_hand_2"   file="../assets/panda/assets/hand_2.obj"/>
    <mesh name="p_hand_3"   file="../assets/panda/assets/hand_3.obj"/>
    <mesh name="p_hand_4"   file="../assets/panda/assets/hand_4.obj"/>
    <mesh name="p_finger_0" file="../assets/panda/assets/finger_0.obj"/>
    <mesh name="p_finger_1" file="../assets/panda/assets/finger_1.obj"/>
  </asset>

  <!-- ── World ───────────────────────────────────────────────────────── -->
  <worldbody>
    <!-- Key light: angled sun from front-left, casts shadows -->
    <light name="sun" directional="true" pos="-1 -3 5" dir="0.15 0.5 -1"
           diffuse="0.80 0.78 0.72" specular="0.25 0.25 0.20" castshadow="true"/>
    <!-- Fill light: soft from right, no shadow -->
    <light name="fill" directional="true" pos="3 2 4" dir="-0.4 -0.3 -1"
           diffuse="0.30 0.35 0.42" specular="0.05 0.05 0.05" castshadow="false"/>
    <!-- Rim light: subtle back-light to separate robot from background -->
    <light name="rim" directional="true" pos="-2 1 3" dir="0.5 -0.2 -1"
           diffuse="0.15 0.18 0.22" specular="0.0 0.0 0.0" castshadow="false"/>

    <geom name="floor" type="plane" size="8 8 0.1" material="floor_mat"
          condim="3" friction="0.8 0.02 0.01"/>

    <!-- ── Go2 robot body ─────────────────────────────────────────── -->
    <body name="base_link" pos="0 0 0.445" childclass="go2">
      <inertial pos="0.021112 0 -0.005366"
                quat="-0.000543471 0.713435 -0.00173769 0.700719"
                mass="6.921" diaginertia="0.107027 0.0980771 0.0244531"/>
      <freejoint name="root"/>

      <!-- Trunk visual -->
      <geom mesh="base_0" material="black"  class="go2_visual"/>
      <geom mesh="base_1" material="black"  class="go2_visual"/>
      <geom mesh="base_2" material="black"  class="go2_visual"/>
      <geom mesh="base_3" material="white"  class="go2_visual"/>
      <geom mesh="base_4" material="gray"   class="go2_visual"/>
      <!-- Trunk collision -->
      <geom size="0.1881 0.04675 0.057" type="box" class="go2_collision"/>
      <geom size="0.05 0.045" pos="0.285 0 0.01" type="cylinder" class="go2_collision"/>
      <geom size="0.047" pos="0.293 0 -0.06" class="go2_collision"/>
      <site name="imu" pos="-0.02557 0 0.04232"/>

      <!-- ── FL leg ──────────────────────────────────────────────── -->
      <body name="FL_hip" pos="0.1934 0.0465 0">
        <inertial pos="-0.0054 0.00194 -0.000105"
                  quat="0.497014 0.499245 0.505462 0.498237"
                  mass="0.678" diaginertia="0.00088403 0.000596003 0.000479967"/>
        <joint name="FL_hip_joint" class="abduction"/>
        <geom mesh="hip_0" material="metal" class="go2_visual"/>
        <geom mesh="hip_1" material="gray"  class="go2_visual"/>
        <geom size="0.046 0.02" pos="0 0.08 0" quat="1 1 0 0" type="cylinder" class="go2_collision"/>
        <body name="FL_thigh" pos="0 0.0955 0">
          <inertial pos="-0.00374 -0.0223 -0.0327"
                    quat="0.829533 0.0847635 -0.0200632 0.551623"
                    mass="1.152" diaginertia="0.00594973 0.00584149 0.000878787"/>
          <joint name="FL_thigh_joint" class="front_hip"/>
          <geom mesh="thigh_0" material="metal" class="go2_visual"/>
          <geom mesh="thigh_1" material="gray"  class="go2_visual"/>
          <geom size="0.1065 0.01225 0.017" pos="0 0 -0.1065"
                quat="0.707107 0 0.707107 0" type="box" class="go2_collision"/>
          <body name="FL_calf" pos="0 0 -0.213">
            <inertial pos="0.00629595 -0.000622121 -0.141417"
                      quat="0.710672 0.00154099 -0.00450087 0.703508"
                      mass="0.241352" diaginertia="0.0014901 0.00146356 5.31397e-05"/>
            <joint name="FL_calf_joint" class="knee"/>
            <geom mesh="calf_0" material="gray"  class="go2_visual"/>
            <geom mesh="calf_1" material="black" class="go2_visual"/>
            <geom size="0.012 0.06" pos="0.008 0 -0.06" quat="0.994493 0 -0.104807 0"
                  type="cylinder" class="go2_collision"/>
            <geom size="0.011 0.0325" pos="0.02 0 -0.148" quat="0.999688 0 0.0249974 0"
                  type="cylinder" class="go2_collision"/>
            <geom pos="0 0 -0.213" mesh="go2_foot" class="go2_visual" material="black"/>
            <geom name="FL_foot_coll" class="foot"/>
            <body name="FL_foot" pos="0 0 -0.213"/>
          </body>
        </body>
      </body>

      <!-- ── FR leg ──────────────────────────────────────────────── -->
      <body name="FR_hip" pos="0.1934 -0.0465 0">
        <inertial pos="-0.0054 -0.00194 -0.000105"
                  quat="0.498237 0.505462 0.499245 0.497014"
                  mass="0.678" diaginertia="0.00088403 0.000596003 0.000479967"/>
        <joint name="FR_hip_joint" class="abduction"/>
        <geom mesh="hip_0" material="metal" class="go2_visual" quat="4.63268e-05 1 0 0"/>
        <geom mesh="hip_1" material="gray"  class="go2_visual" quat="4.63268e-05 1 0 0"/>
        <geom size="0.046 0.02" pos="0 -0.08 0" quat="0.707107 0.707107 0 0"
              type="cylinder" class="go2_collision"/>
        <body name="FR_thigh" pos="0 -0.0955 0">
          <inertial pos="-0.00374 0.0223 -0.0327"
                    quat="0.551623 -0.0200632 0.0847635 0.829533"
                    mass="1.152" diaginertia="0.00594973 0.00584149 0.000878787"/>
          <joint name="FR_thigh_joint" class="front_hip"/>
          <geom mesh="thigh_mirror_0" material="metal" class="go2_visual"/>
          <geom mesh="thigh_mirror_1" material="gray"  class="go2_visual"/>
          <geom size="0.1065 0.01225 0.017" pos="0 0 -0.1065"
                quat="0.707107 0 0.707107 0" type="box" class="go2_collision"/>
          <body name="FR_calf" pos="0 0 -0.213">
            <inertial pos="0.00629595 0.000622121 -0.141417"
                      quat="0.703508 -0.00450087 0.00154099 0.710672"
                      mass="0.241352" diaginertia="0.0014901 0.00146356 5.31397e-05"/>
            <joint name="FR_calf_joint" class="knee"/>
            <geom mesh="calf_mirror_0" material="gray"  class="go2_visual"/>
            <geom mesh="calf_mirror_1" material="black" class="go2_visual"/>
            <geom size="0.013 0.06" pos="0.01 0 -0.06" quat="0.995004 0 -0.0998334 0"
                  type="cylinder" class="go2_collision"/>
            <geom size="0.011 0.0325" pos="0.02 0 -0.148" quat="0.999688 0 0.0249974 0"
                  type="cylinder" class="go2_collision"/>
            <geom pos="0 0 -0.213" mesh="go2_foot" class="go2_visual" material="black"/>
            <geom name="FR_foot_coll" class="foot"/>
            <body name="FR_foot" pos="0 0 -0.213"/>
          </body>
        </body>
      </body>

      <!-- ── RL leg ──────────────────────────────────────────────── -->
      <body name="RL_hip" pos="-0.1934 0.0465 0">
        <inertial pos="0.0054 0.00194 -0.000105"
                  quat="0.505462 0.498237 0.497014 0.499245"
                  mass="0.678" diaginertia="0.00088403 0.000596003 0.000479967"/>
        <joint name="RL_hip_joint" class="abduction"/>
        <geom mesh="hip_0" material="metal" class="go2_visual" quat="4.63268e-05 0 1 0"/>
        <geom mesh="hip_1" material="gray"  class="go2_visual" quat="4.63268e-05 0 1 0"/>
        <geom size="0.046 0.02" pos="0 0.08 0" quat="0.707107 0.707107 0 0"
              type="cylinder" class="go2_collision"/>
        <body name="RL_thigh" pos="0 0.0955 0">
          <inertial pos="-0.00374 -0.0223 -0.0327"
                    quat="0.829533 0.0847635 -0.0200632 0.551623"
                    mass="1.152" diaginertia="0.00594973 0.00584149 0.000878787"/>
          <joint name="RL_thigh_joint" class="back_hip"/>
          <geom mesh="thigh_0" material="metal" class="go2_visual"/>
          <geom mesh="thigh_1" material="gray"  class="go2_visual"/>
          <geom size="0.1065 0.01225 0.017" pos="0 0 -0.1065"
                quat="0.707107 0 0.707107 0" type="box" class="go2_collision"/>
          <body name="RL_calf" pos="0 0 -0.213">
            <inertial pos="0.00629595 -0.000622121 -0.141417"
                      quat="0.710672 0.00154099 -0.00450087 0.703508"
                      mass="0.241352" diaginertia="0.0014901 0.00146356 5.31397e-05"/>
            <joint name="RL_calf_joint" class="knee"/>
            <geom mesh="calf_0" material="gray"  class="go2_visual"/>
            <geom mesh="calf_1" material="black" class="go2_visual"/>
            <geom size="0.013 0.06" pos="0.01 0 -0.06" quat="0.995004 0 -0.0998334 0"
                  type="cylinder" class="go2_collision"/>
            <geom size="0.011 0.0325" pos="0.02 0 -0.148" quat="0.999688 0 0.0249974 0"
                  type="cylinder" class="go2_collision"/>
            <geom pos="0 0 -0.213" mesh="go2_foot" class="go2_visual" material="black"/>
            <geom name="RL_foot_coll" class="foot"/>
            <body name="RL_foot" pos="0 0 -0.213"/>
          </body>
        </body>
      </body>

      <!-- ── RR leg ──────────────────────────────────────────────── -->
      <body name="RR_hip" pos="-0.1934 -0.0465 0">
        <inertial pos="0.0054 -0.00194 -0.000105"
                  quat="0.499245 0.497014 0.498237 0.505462"
                  mass="0.678" diaginertia="0.00088403 0.000596003 0.000479967"/>
        <joint name="RR_hip_joint" class="abduction"/>
        <geom mesh="hip_0" material="metal" class="go2_visual"
              quat="2.14617e-09 4.63268e-05 4.63268e-05 -1"/>
        <geom mesh="hip_1" material="gray"  class="go2_visual"
              quat="2.14617e-09 4.63268e-05 4.63268e-05 -1"/>
        <geom size="0.046 0.02" pos="0 -0.08 0" quat="0.707107 0.707107 0 0"
              type="cylinder" class="go2_collision"/>
        <body name="RR_thigh" pos="0 -0.0955 0">
          <inertial pos="-0.00374 0.0223 -0.0327"
                    quat="0.551623 -0.0200632 0.0847635 0.829533"
                    mass="1.152" diaginertia="0.00594973 0.00584149 0.000878787"/>
          <joint name="RR_thigh_joint" class="back_hip"/>
          <geom mesh="thigh_mirror_0" material="metal" class="go2_visual"/>
          <geom mesh="thigh_mirror_1" material="gray"  class="go2_visual"/>
          <geom size="0.1065 0.01225 0.017" pos="0 0 -0.1065"
                quat="0.707107 0 0.707107 0" type="box" class="go2_collision"/>
          <body name="RR_calf" pos="0 0 -0.213">
            <inertial pos="0.00629595 0.000622121 -0.141417"
                      quat="0.703508 -0.00450087 0.00154099 0.710672"
                      mass="0.241352" diaginertia="0.0014901 0.00146356 5.31397e-05"/>
            <joint name="RR_calf_joint" class="knee"/>
            <geom mesh="calf_mirror_0" material="gray"  class="go2_visual"/>
            <geom mesh="calf_mirror_1" material="black" class="go2_visual"/>
            <geom size="0.013 0.06" pos="0.01 0 -0.06" quat="0.995004 0 -0.0998334 0"
                  type="cylinder" class="go2_collision"/>
            <geom size="0.011 0.0325" pos="0.02 0 -0.148" quat="0.999688 0 0.0249974 0"
                  type="cylinder" class="go2_collision"/>
            <geom pos="0 0 -0.213" mesh="go2_foot" class="go2_visual" material="black"/>
            <geom name="RR_foot_coll" class="foot"/>
            <body name="RR_foot" pos="0 0 -0.213"/>
          </body>
        </body>
      </body>

      <!-- ── Franka Panda arm (rigidly mounted on trunk) ─────────── -->
      <!--
        Mounting geometry:
          base_link origin: robot body center (~0.445 m above ground at rest)
          Trunk box half-size Z: 0.057 m → trunk top at +0.057 m from base_link
          Mount offset: 0.10 m above base_link (safely above trunk top)
          No joint → rigid attachment
          childclass="panda" overrides inherited "go2" defaults for arm bodies
      -->
      <body name="panda_link0" pos="0 0 0.10" childclass="panda">
        <!-- Inertial: original × 0.35 -->
        <inertial mass="0.220419" pos="-0.041018 -0.00014 0.049974"
          fullinertia="0.001103 0.001358 0.001500 2.90e-7 5.25e-5 2.88e-6"/>
        <geom mesh="p_link0_0"  material="p_off_white" class="panda_visual"/>
        <geom mesh="p_link0_1"  material="p_dark"      class="panda_visual"/>
        <geom mesh="p_link0_2"  material="p_off_white" class="panda_visual"/>
        <geom mesh="p_link0_3"  material="p_dark"      class="panda_visual"/>
        <geom mesh="p_link0_4"  material="p_off_white" class="panda_visual"/>
        <geom mesh="p_link0_5"  material="p_dark"      class="panda_visual"/>
        <geom mesh="p_link0_7"  material="white"        class="panda_visual"/>
        <geom mesh="p_link0_8"  material="white"        class="panda_visual"/>
        <geom mesh="p_link0_9"  material="p_dark"      class="panda_visual"/>
        <geom mesh="p_link0_10" material="p_off_white" class="panda_visual"/>
        <geom mesh="p_link0_11" material="white"        class="panda_visual"/>
        <geom mesh="p_link0_c"  class="panda_collision"/>

        <body name="panda_link1" pos="0 0 0.333">
          <inertial mass="1.739739" pos="0.003875 0.002081 -0.04762"
            fullinertia="0.24618 0.24731 0.003191 -4.865e-5 0.002370 0.006709"/>
          <joint name="joint1" class="panda"/>
          <geom material="white" mesh="p_link1" class="panda_visual"/>
          <geom mesh="p_link1_c" class="panda_collision"/>

          <body name="panda_link2" quat="1 -1 0 0">
            <inertial mass="0.226424" pos="-0.003141 -0.02872 0.003495"
              fullinertia="0.002787 0.009839 0.009099 -0.001374 0.003589 2.464e-4"/>
            <joint name="joint2" class="panda" range="-1.7628 1.7628"/>
            <geom material="white" mesh="p_link2" class="panda_visual"/>
            <geom mesh="p_link2_c" class="panda_collision"/>

            <body name="panda_link3" pos="0 -0.316 0" quat="1 1 0 0">
              <inertial mass="1.130011" pos="2.7518e-2 3.9252e-2 -6.6502e-2"
                fullinertia="0.013035 0.012654 0.003791 -0.001666 -0.003989 -0.004482"/>
              <joint name="joint3" class="panda"/>
              <geom mesh="p_link3_0" material="white"      class="panda_visual"/>
              <geom mesh="p_link3_1" material="white"      class="panda_visual"/>
              <geom mesh="p_link3_2" material="white"      class="panda_visual"/>
              <geom mesh="p_link3_3" material="p_dark"    class="panda_visual"/>
              <geom mesh="p_link3_c" class="panda_collision"/>

              <body name="panda_link4" pos="0.0825 0 0" quat="1 1 0 0">
                <inertial mass="1.255763" pos="-5.317e-2 1.04419e-1 2.7454e-2"
                  fullinertia="0.009049 0.006843 0.009913 0.002729 -4.662e-4 0.003024"/>
                <joint name="joint4" class="panda" range="-3.0718 -0.0698"/>
                <geom mesh="p_link4_0" material="white"      class="panda_visual"/>
                <geom mesh="p_link4_1" material="white"      class="panda_visual"/>
                <geom mesh="p_link4_2" material="p_dark"    class="panda_visual"/>
                <geom mesh="p_link4_3" material="white"      class="panda_visual"/>
                <geom mesh="p_link4_c" class="panda_collision"/>

                <body name="panda_link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                  <inertial mass="0.429081" pos="-1.1953e-2 4.1065e-2 -3.8437e-2"
                    fullinertia="0.012442 0.010316 0.003019 -7.410e-4 -0.001413 8.015e-5"/>
                  <joint name="joint5" class="panda"/>
                  <geom mesh="p_link5_0" material="p_dark"        class="panda_visual"/>
                  <geom mesh="p_link5_1" material="white"          class="panda_visual"/>
                  <geom mesh="p_link5_2" material="white"          class="panda_visual"/>
                  <geom mesh="p_link5_c0" class="panda_collision"/>
                  <geom mesh="p_link5_c1" class="panda_collision"/>
                  <geom mesh="p_link5_c2" class="panda_collision"/>

                  <body name="panda_link6" quat="1 1 0 0">
                    <inertial mass="0.583294" pos="6.0149e-2 -1.4117e-2 -1.0517e-2"
                      fullinertia="6.874e-4 1.524e-3 1.902e-3 3.815e-5 -4.053e-4 1.194e-4"/>
                    <joint name="joint6" class="panda" range="-0.0175 3.7525"/>
                    <geom mesh="p_link6_0"  material="p_off_white"  class="panda_visual"/>
                    <geom mesh="p_link6_1"  material="white"         class="panda_visual"/>
                    <geom mesh="p_link6_2"  material="p_dark"       class="panda_visual"/>
                    <geom mesh="p_link6_3"  material="white"         class="panda_visual"/>
                    <geom mesh="p_link6_4"  material="white"         class="panda_visual"/>
                    <geom mesh="p_link6_5"  material="white"         class="panda_visual"/>
                    <geom mesh="p_link6_6"  material="white"         class="panda_visual"/>
                    <geom mesh="p_link6_7"  material="p_light_blue" class="panda_visual"/>
                    <geom mesh="p_link6_8"  material="p_light_blue" class="panda_visual"/>
                    <geom mesh="p_link6_9"  material="p_dark"       class="panda_visual"/>
                    <geom mesh="p_link6_10" material="p_dark"       class="panda_visual"/>
                    <geom mesh="p_link6_11" material="white"         class="panda_visual"/>
                    <geom mesh="p_link6_12" material="p_green"      class="panda_visual"/>
                    <geom mesh="p_link6_13" material="white"         class="panda_visual"/>
                    <geom mesh="p_link6_14" material="p_dark"       class="panda_visual"/>
                    <geom mesh="p_link6_15" material="p_dark"       class="panda_visual"/>
                    <geom mesh="p_link6_16" material="white"         class="panda_visual"/>
                    <geom mesh="p_link6_c"  class="panda_collision"/>

                    <body name="panda_link7" pos="0.088 0 0" quat="1 1 0 0">
                      <inertial mass="0.257433" pos="1.0517e-2 -4.252e-3 6.1597e-2"
                        fullinertia="4.381e-3 3.509e-3 1.685e-3 -1.498e-4 -4.186e-4 -2.594e-4"/>
                      <joint name="joint7" class="panda"/>
                      <geom mesh="p_link7_0" material="white"  class="panda_visual"/>
                      <geom mesh="p_link7_1" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_2" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_3" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_4" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_5" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_6" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_7" material="white"  class="panda_visual"/>
                      <geom mesh="p_link7_c" class="panda_collision"/>

                      <body name="panda_hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                        <inertial mass="0.2555" pos="-0.01 0 0.03"
                                  diaginertia="3.5e-4 8.75e-4 5.95e-4"/>
                        <geom mesh="p_hand_0" material="p_off_white" class="panda_visual"/>
                        <geom mesh="p_hand_1" material="p_dark"      class="panda_visual"/>
                        <geom mesh="p_hand_2" material="p_dark"      class="panda_visual"/>
                        <geom mesh="p_hand_3" material="white"        class="panda_visual"/>
                        <geom mesh="p_hand_4" material="p_off_white" class="panda_visual"/>
                        <geom mesh="p_hand_c" class="panda_collision"/>

                        <!-- End-effector site: between fingertips -->
                        <site name="ee_site" pos="0 0 0.12" size="0.01" rgba="0 1 0 0.5"/>

                        <body name="panda_left_finger" pos="0 0 0.0584">
                          <inertial mass="0.00525" pos="0 0 0"
                                    diaginertia="8.313e-7 8.313e-7 2.625e-7"/>
                          <joint name="finger_joint1" class="panda_finger"/>
                          <geom mesh="p_finger_0" material="p_off_white" class="panda_visual"/>
                          <geom mesh="p_finger_1" material="p_dark"      class="panda_visual"/>
                          <geom mesh="p_finger_0" class="panda_collision"/>
                          <geom class="ftp1"/>
                          <geom class="ftp2"/>
                          <geom class="ftp3"/>
                          <geom class="ftp4"/>
                          <geom class="ftp5"/>
                        </body>

                        <body name="panda_right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                          <inertial mass="0.00525" pos="0 0 0"
                                    diaginertia="8.313e-7 8.313e-7 2.625e-7"/>
                          <joint name="finger_joint2" class="panda_finger"/>
                          <geom mesh="p_finger_0" material="p_off_white" class="panda_visual"/>
                          <geom mesh="p_finger_1" material="p_dark"      class="panda_visual"/>
                          <geom mesh="p_finger_0" class="panda_collision"/>
                          <geom class="ftp1"/>
                          <geom class="ftp2"/>
                          <geom class="ftp3"/>
                          <geom class="ftp4"/>
                          <geom class="ftp5"/>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

    </body><!-- end base_link -->

    <!-- Worktable: fixed to world at arm-reachable height in front of Go2 -->
    <!-- Top surface at z = body_pos(0) + geom_pos(0.285) + half-sz(0.015) = 0.30 m -->
    <!-- Near edge at x = 1.6 - 0.25 = 1.35 m; robot front stops at ~1.28 m (safe) -->
    <!-- Y half-size 0.35 m → table spans y=[-0.35, +0.35], wide enough for pick+place -->
    <body name="worktable" pos="1.6 0 0">
      <geom name="table_top" type="box" size="0.25 0.35 0.015"
            pos="0 0 0.285" material="table_mat" condim="6" friction="0.8 0.02 0.01"/>
      <!-- Legs: center at z=0.1425, half-height 0.1425 — spans z=[0, 0.285] m -->
      <geom name="tleg_fl" type="cylinder" size="0.018 0.1425" pos=" 0.22  0.30 0.1425" material="table_leg_mat"/>
      <geom name="tleg_fr" type="cylinder" size="0.018 0.1425" pos=" 0.22 -0.30 0.1425" material="table_leg_mat"/>
      <geom name="tleg_rl" type="cylinder" size="0.018 0.1425" pos="-0.22  0.30 0.1425" material="table_leg_mat"/>
      <geom name="tleg_rr" type="cylinder" size="0.018 0.1425" pos="-0.22 -0.30 0.1425" material="table_leg_mat"/>
      <site name="table_center" pos="0 0 0.30" size="0.01"/>
    </body>

    <!-- Placement plate (bright green target zone on table surface) -->
    <!-- Plate top at z = 0.30 + 0.006 = 0.306 m                    -->
    <!-- Cube resting on plate: center at 0.306 + 0.025 = 0.331 m   -->
    <body name="target_plate" pos="1.6 0.20 0.30">
      <geom name="plate_geom" type="box" size="0.09 0.07 0.006"
            material="plate_mat" condim="6" friction="0.8 0.02 0.01"/>
      <site name="plate_center" pos="0 0 0.006" size="0.015" rgba="0.8 1 0.8 1"/>
    </body>

    <!-- Pickup cube (vivid red, free to move) resting on worktable  -->
    <!-- Cube center at table_top(0.30) + cube_half(0.025) = 0.325 m -->
    <!-- Listed last so cube_joint freejoint stays at qpos[28:35],   -->
    <!-- keeping base_link freejoint at qpos[0:7] as expected.        -->
    <!-- Inertia: solid box, mass=0.1, side=0.05 m, I=m/6*a²=4.17e-5 -->
    <body name="target_cube" pos="1.6 0 0.325">
      <freejoint name="cube_joint"/>
      <inertial mass="0.1" pos="0 0 0" diaginertia="4.17e-5 4.17e-5 4.17e-5"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025"
            material="cube_mat" condim="6" friction="1.5 0.05 0.01"
            solref="0.002 1" solimp="0.9 0.95 0.001"/>
      <site name="cube_center" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>

  <!-- ── Actuators ───────────────────────────────────────────────────── -->
  <actuator>
    <!-- Go2 leg motors: torque control (PD computed externally in controller) -->
    <!-- Actuator order: FR, FL, RR, RL (matches original unitree_mujoco) -->
    <motor class="abduction" name="FR_hip"   joint="FR_hip_joint"/>
    <motor class="front_hip" name="FR_thigh" joint="FR_thigh_joint"/>
    <motor class="knee"      name="FR_calf"  joint="FR_calf_joint"/>
    <motor class="abduction" name="FL_hip"   joint="FL_hip_joint"/>
    <motor class="front_hip" name="FL_thigh" joint="FL_thigh_joint"/>
    <motor class="knee"      name="FL_calf"  joint="FL_calf_joint"/>
    <motor class="abduction" name="RR_hip"   joint="RR_hip_joint"/>
    <motor class="back_hip"  name="RR_thigh" joint="RR_thigh_joint"/>
    <motor class="knee"      name="RR_calf"  joint="RR_calf_joint"/>
    <motor class="abduction" name="RL_hip"   joint="RL_hip_joint"/>
    <motor class="back_hip"  name="RL_thigh" joint="RL_thigh_joint"/>
    <motor class="knee"      name="RL_calf"  joint="RL_calf_joint"/>

    <!-- Panda arm: position control with integrated PD (kp/kd baked in) -->
    <!-- ctrl input = desired joint angle (rad), gripper ctrl = 0-255 mapped -->
    <general class="panda" name="actuator1" joint="joint1"
             gainprm="4500" biasprm="0 -4500 -450"/>
    <general class="panda" name="actuator2" joint="joint2"
             gainprm="4500" biasprm="0 -4500 -450" ctrlrange="-1.7628 1.7628"/>
    <general class="panda" name="actuator3" joint="joint3"
             gainprm="3500" biasprm="0 -3500 -350"/>
    <general class="panda" name="actuator4" joint="joint4"
             gainprm="3500" biasprm="0 -3500 -350" ctrlrange="-3.0718 -0.0698"/>
    <general class="panda" name="actuator5" joint="joint5"
             gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <general class="panda" name="actuator6" joint="joint6"
             gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"
             ctrlrange="-0.0175 3.7525"/>
    <general class="panda" name="actuator7" joint="joint7"
             gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <!-- Gripper: tendon-based, ctrl range 0-255 mapped to 0-0.04 m -->
    <general class="panda" name="actuator8" tendon="panda_split"
             forcerange="-100 100" ctrlrange="0 255"
             gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>
  </actuator>

  <!-- ── Tendons ──────────────────────────────────────────────────────── -->
  <tendon>
    <!-- Finger synchronization: both fingers move together -->
    <fixed name="panda_split">
      <joint joint="finger_joint1" coef="0.5"/>
      <joint joint="finger_joint2" coef="0.5"/>
    </fixed>
  </tendon>

  <!-- ── Equality constraints ─────────────────────────────────────────── -->
  <equality>
    <!-- Finger mimic: right finger mirrors left finger -->
    <joint joint1="finger_joint1" joint2="finger_joint2"
           solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>

  <!-- ── Sensors ──────────────────────────────────────────────────────── -->
  <sensor>
    <!-- Go2 joint position sensors (12) -->
    <jointpos name="FR_hip_pos"   joint="FR_hip_joint"/>
    <jointpos name="FR_thigh_pos" joint="FR_thigh_joint"/>
    <jointpos name="FR_calf_pos"  joint="FR_calf_joint"/>
    <jointpos name="FL_hip_pos"   joint="FL_hip_joint"/>
    <jointpos name="FL_thigh_pos" joint="FL_thigh_joint"/>
    <jointpos name="FL_calf_pos"  joint="FL_calf_joint"/>
    <jointpos name="RR_hip_pos"   joint="RR_hip_joint"/>
    <jointpos name="RR_thigh_pos" joint="RR_thigh_joint"/>
    <jointpos name="RR_calf_pos"  joint="RR_calf_joint"/>
    <jointpos name="RL_hip_pos"   joint="RL_hip_joint"/>
    <jointpos name="RL_thigh_pos" joint="RL_thigh_joint"/>
    <jointpos name="RL_calf_pos"  joint="RL_calf_joint"/>

    <!-- Go2 joint velocity sensors (12) -->
    <jointvel name="FR_hip_vel"   joint="FR_hip_joint"/>
    <jointvel name="FR_thigh_vel" joint="FR_thigh_joint"/>
    <jointvel name="FR_calf_vel"  joint="FR_calf_joint"/>
    <jointvel name="FL_hip_vel"   joint="FL_hip_joint"/>
    <jointvel name="FL_thigh_vel" joint="FL_thigh_joint"/>
    <jointvel name="FL_calf_vel"  joint="FL_calf_joint"/>
    <jointvel name="RR_hip_vel"   joint="RR_hip_joint"/>
    <jointvel name="RR_thigh_vel" joint="RR_thigh_joint"/>
    <jointvel name="RR_calf_vel"  joint="RR_calf_joint"/>
    <jointvel name="RL_hip_vel"   joint="RL_hip_joint"/>
    <jointvel name="RL_thigh_vel" joint="RL_thigh_joint"/>
    <jointvel name="RL_calf_vel"  joint="RL_calf_joint"/>

    <!-- Go2 joint torque sensors (12) -->
    <jointactuatorfrc name="FR_hip_torque"   joint="FR_hip_joint"   noise="0.01"/>
    <jointactuatorfrc name="FR_thigh_torque" joint="FR_thigh_joint" noise="0.01"/>
    <jointactuatorfrc name="FR_calf_torque"  joint="FR_calf_joint"  noise="0.01"/>
    <jointactuatorfrc name="FL_hip_torque"   joint="FL_hip_joint"   noise="0.01"/>
    <jointactuatorfrc name="FL_thigh_torque" joint="FL_thigh_joint" noise="0.01"/>
    <jointactuatorfrc name="FL_calf_torque"  joint="FL_calf_joint"  noise="0.01"/>
    <jointactuatorfrc name="RR_hip_torque"   joint="RR_hip_joint"   noise="0.01"/>
    <jointactuatorfrc name="RR_thigh_torque" joint="RR_thigh_joint" noise="0.01"/>
    <jointactuatorfrc name="RR_calf_torque"  joint="RR_calf_joint"  noise="0.01"/>
    <jointactuatorfrc name="RL_hip_torque"   joint="RL_hip_joint"   noise="0.01"/>
    <jointactuatorfrc name="RL_thigh_torque" joint="RL_thigh_joint" noise="0.01"/>
    <jointactuatorfrc name="RL_calf_torque"  joint="RL_calf_joint"  noise="0.01"/>

    <!-- Go2 IMU -->
    <framequat    name="imu_quat" objtype="site" objname="imu"/>
    <gyro         name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
    <framepos     name="frame_pos" objtype="site" objname="imu"/>
    <framelinvel  name="frame_vel" objtype="site" objname="imu"/>

    <!-- Panda end-effector tracking -->
    <framepos  name="ee_pos"  objtype="site" objname="ee_site"/>
    <framequat name="ee_quat" objtype="site" objname="ee_site"/>

    <!-- Cube position tracking -->
    <framepos name="cube_pos" objtype="site" objname="cube_center"/>
  </sensor>

  <!-- ── Contact exclusions ───────────────────────────────────────────── -->
  <contact>
    <!-- Panda adjacent-link exclusions (prevent self-collision artifacts) -->
    <exclude body1="panda_link0" body2="panda_link1"/>
    <exclude body1="panda_link1" body2="panda_link2"/>
    <exclude body1="panda_link2" body2="panda_link3"/>
    <exclude body1="panda_link3" body2="panda_link4"/>
    <exclude body1="panda_link4" body2="panda_link5"/>
    <exclude body1="panda_link5" body2="panda_link6"/>
    <exclude body1="panda_link6" body2="panda_link7"/>
    <exclude body1="panda_link7" body2="panda_hand"/>
    <!-- Panda-Go2 trunk exclusion (mounting interface) -->
    <exclude body1="base_link"   body2="panda_link0"/>
  </contact>

  <!-- ── Keyframe ─────────────────────────────────────────────────────── -->
  <keyframe>
    <!--
      qpos layout (35 total):
        [0:3]   base_link position (x, y, z)
        [3:7]   base_link quaternion (w, x, y, z)
        [7:19]  leg joints FR/FL/RR/RL × [hip, thigh, calf]
        [19:26] Panda joint1-7
        [26:28] finger_joint1, finger_joint2
        [28:35] target_cube freejoint (x, y, z, qw, qx, qy, qz)
      ctrl layout (20 total):
        [0:12]  Go2 leg motors (FR, FL, RR, RL order)
        [12:19] Panda arm actuators 1-7
        [19]    Gripper actuator8
    -->
    <key name="home"
      qpos="0 0 0.27 1 0 0 0
            0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8
            0 0 0 -1.5708 0 1.5708 -0.7853
            0.04 0.04
            1.6 0 0.325 1 0 0 0"
      ctrl="0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8
            0 0 0 -1.5708 0 1.5708 -0.7853  255"/>
  </keyframe>

</mujoco>
"""
    return xml


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "combined.xml"
    out_path.write_text(build_combined_xml(), encoding="utf-8")
    print(f"Written: {out_path}")
    print("Verifying model loads...")

    import mujoco  # noqa: PLC0415
    try:
        m = mujoco.MjModel.from_xml_path(str(out_path))
        d = mujoco.MjData(m)
        total_mass = sum(m.body_mass)
        print(f"  nq={m.nq}  nv={m.nv}  nu={m.nu}  nbody={m.nbody}")
        print(f"  Total mass: {total_mass:.2f} kg")
        print(f"  Sensor data size: {m.nsensordata}")
        print("  [OK] Model loaded successfully!")
    except Exception as exc:
        print(f"  [ERR] {exc}")
        raise


if __name__ == "__main__":
    main()
