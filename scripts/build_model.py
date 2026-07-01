"""
Build the combined Go2 + arm MJCF model.

Reads source XML files from assets/, applies:
 - Mass scaling on the arm to match Go2 payload capacity
 - Boosted Go2 motor limits for heavier payload
 - Merged default classes with unambiguous names
 - Explicit mesh paths relative to models/ directory
 - Target cube and end-effector site
Writes models/combined.xml.

Arm and end-effector are both swappable (see controllers/arms.py and
controllers/end_effectors.py); this module dispatches on ArmSpec.actuator_kind
("general_pd" -> Franka path, "position_pd" -> Kinova path) and on
EndEffectorSpec.name to build the right XML subtrees.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR))

# ── Scene visual presets ────────────────────────────────────────────────────
# Each preset is a flat dict of named visual parameters.  The keys map 1-to-1
# onto the XML attributes they replace in build_combined_xml().
# "default" preserves the original bright studio look exactly.
SCENE_PRESETS: dict[str, dict] = {
    "default": {
        "skybox_rgb1": ".45 .60 .80",
        "skybox_rgb2": ".08 .10 .18",
        "headlight_ambient": "0.35 0.35 0.38",
        "headlight_diffuse": "0.7 0.7 0.7",
        "haze": "0.18 0.22 0.30 1",
        "fogstart": "8",
        "fogend": "20",
        "shadowscale": "0.6",
        "sun_pos": "-1 -3 5",
        "sun_dir": "0.15 0.5 -1",
        "sun_diffuse": "0.80 0.78 0.72",
        "sun_specular": "0.25 0.25 0.20",
        "fill_pos": "3 2 4",
        "fill_dir": "-0.4 -0.3 -1",
        "fill_diffuse": "0.30 0.35 0.42",
        "rim_pos": "-2 1 3",
        "rim_dir": "0.5 -0.2 -1",
        "rim_diffuse": "0.15 0.18 0.22",
        "floor_rgb1": ".82 .82 .82",
        "floor_rgb2": ".65 .65 .65",
    },
    "warehouse": {
        # Industrial storage: warm sodium-vapor overhead, dark concrete, deep shadows.
        "skybox_rgb1": "0.08 0.07 0.06",
        "skybox_rgb2": "0.04 0.04 0.03",
        "headlight_ambient": "0.08 0.07 0.06",
        "headlight_diffuse": "0.25 0.22 0.18",
        "haze": "0.20 0.18 0.14 1",
        "fogstart": "6",
        "fogend": "15",
        "shadowscale": "0.30",
        "sun_pos": "-1 -3 5",
        "sun_dir": "0.15 0.5 -1",
        "sun_diffuse": "0.85 0.70 0.45",
        "sun_specular": "0.15 0.10 0.05",
        "fill_pos": "3 2 4",
        "fill_dir": "-0.4 -0.3 -1",
        "fill_diffuse": "0.10 0.09 0.08",
        "rim_pos": "-2 1 3",
        "rim_dir": "0.5 -0.2 -1",
        "rim_diffuse": "0.05 0.05 0.04",
        "floor_rgb1": "0.30 0.28 0.25",
        "floor_rgb2": "0.24 0.23 0.20",
    },
    "lab": {
        # Research lab: cool fluorescent overhead, neutral gray floor, crisp controlled light.
        "skybox_rgb1": "0.06 0.06 0.08",
        "skybox_rgb2": "0.03 0.03 0.04",
        "headlight_ambient": "0.18 0.20 0.24",
        "headlight_diffuse": "0.40 0.42 0.50",
        "haze": "0.08 0.10 0.14 1",
        "fogstart": "12",
        "fogend": "25",
        "shadowscale": "0.45",
        "sun_pos": "0 0 6",
        "sun_dir": "0.05 0.1 -1",
        "sun_diffuse": "0.72 0.76 0.88",
        "sun_specular": "0.10 0.10 0.14",
        "fill_pos": "3 2 4",
        "fill_dir": "-0.4 -0.3 -1",
        "fill_diffuse": "0.12 0.15 0.20",
        "rim_pos": "-2 1 3",
        "rim_dir": "0.5 -0.2 -1",
        "rim_diffuse": "0.06 0.08 0.12",
        "floor_rgb1": "0.54 0.54 0.56",
        "floor_rgb2": "0.46 0.46 0.48",
    },
    "outdoor": {
        # Golden hour: warm low sun, long crisp shadows, sky gradient, sandy ground.
        "skybox_rgb1": "0.75 0.50 0.22",
        "skybox_rgb2": "0.12 0.22 0.48",
        "headlight_ambient": "0.10 0.09 0.07",
        "headlight_diffuse": "0.12 0.10 0.08",
        "haze": "0.32 0.25 0.16 1",
        "fogstart": "10",
        "fogend": "22",
        "shadowscale": "0.20",
        "sun_pos": "5 -8 4",
        "sun_dir": "-0.3 0.7 -1",
        "sun_diffuse": "0.98 0.75 0.38",
        "sun_specular": "0.20 0.14 0.05",
        "fill_pos": "-4 5 6",
        "fill_dir": "0.3 -0.5 -1",
        "fill_diffuse": "0.15 0.22 0.38",
        "rim_pos": "-3 -2 3",
        "rim_dir": "0.5 0.3 -1",
        "rim_diffuse": "0.18 0.12 0.05",
        "floor_rgb1": "0.65 0.58 0.42",
        "floor_rgb2": "0.55 0.50 0.36",
    },
    "cinematic": {
        # Dark studio: single strong key, near-black shadows, maximum contrast.
        "skybox_rgb1": "0.03 0.04 0.06",
        "skybox_rgb2": "0.01 0.01 0.02",
        "headlight_ambient": "0.04 0.04 0.05",
        "headlight_diffuse": "0.15 0.15 0.18",
        "haze": "0.04 0.05 0.08 1",
        "fogstart": "8",
        "fogend": "18",
        "shadowscale": "0.15",
        "sun_pos": "-3 -5 6",
        "sun_dir": "0.4 0.6 -1",
        "sun_diffuse": "0.95 0.88 0.75",
        "sun_specular": "0.25 0.20 0.15",
        "fill_pos": "4 3 4",
        "fill_dir": "-0.5 -0.4 -1",
        "fill_diffuse": "0.06 0.10 0.18",
        "rim_pos": "3 2 3",
        "rim_dir": "-0.5 -0.3 -1",
        "rim_diffuse": "0.08 0.14 0.22",
        "floor_rgb1": "0.14 0.14 0.16",
        "floor_rgb2": "0.10 0.10 0.12",
    },
}
from controllers.arms import ARMS, ArmSpec, DEFAULT_ARM, get_arm_spec, validate_combo  # noqa: E402
from controllers.end_effectors import (  # noqa: E402
    EndEffectorSpec,
    MountOverride,
    DEFAULT_END_EFFECTOR,
    get_spec,
    get_mount_override,
)


def _scale_inertia_by(value_str: str, scale: float) -> str:
    """Scale a space-separated inertia string by `scale`."""
    parts = value_str.strip().split()
    scaled = [f"{float(p) * scale:.6g}" for p in parts]
    return " ".join(scaled)


def _fmt(*vals: float) -> str:
    """Format floats as a space-separated MJCF attribute string."""
    return " ".join(f"{v:.10g}" for v in vals)


# ─────────────────────────────────────────────────────────────────────────
# Gripper subtrees (end-effector dimension)
# ─────────────────────────────────────────────────────────────────────────

def _franka_gripper_xml(spec: EndEffectorSpec) -> str:
    """Franka stock two-finger gripper -- identical to the original literal block."""
    return f"""                      <body name="panda_hand" pos="{_fmt(*spec.mount_pos)}" quat="{_fmt(*spec.mount_quat)}">
                        <inertial mass="0.2555" pos="-0.01 0 0.03"
                                  diaginertia="3.5e-4 8.75e-4 5.95e-4"/>
                        <geom mesh="p_hand_0" material="p_off_white" class="panda_visual"/>
                        <geom mesh="p_hand_1" material="p_dark"      class="panda_visual"/>
                        <geom mesh="p_hand_2" material="p_dark"      class="panda_visual"/>
                        <geom mesh="p_hand_3" material="white"        class="panda_visual"/>
                        <geom mesh="p_hand_4" material="p_off_white" class="panda_visual"/>
                        <geom mesh="p_hand_c" class="panda_collision"/>

                        <!-- End-effector site: between fingertips -->
                        <site name="ee_site" pos="{_fmt(*spec.ee_site_pos)}" size="0.01" rgba="0 1 0 0.5"/>

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
                      </body>"""


def _robotiq_base_children_xml(spec: EndEffectorSpec) -> str:
    """Everything inside r2f85_base: inertial + 2 geoms + both 4-bar-linkage
    finger subtrees (unchanged from upstream robotiq_2f85/2f85.xml). Mounting
    -- where r2f85_base itself sits, and where ee_site sits -- is handled by
    the caller, since that differs between the default Franka-relative mount
    and Kinova's mount_override.
    """
    s = spec.mass_scale

    def mass(value: float) -> str:
        return f"{value * s:.6g}"

    def di(*values: float) -> str:
        return _scale_inertia_by(" ".join(str(v) for v in values), s)

    return f"""<inertial mass="{mass(0.777441)}" pos="0 -2.70394e-05 0.0354675" quat="1 -0.00152849 0 0"
                            diaginertia="{di(0.000260285, 0.000225381, 0.000152708)}"/>
                          <geom mesh="r2f85_base" material="r2f85_black" class="r2f85_visual"/>
                          <geom mesh="r2f85_base" class="r2f85_collision"/>

                          <!-- Right-hand side 4-bar linkage -->
                          <body name="r2f85_right_driver" pos="0 0.0306011 0.054904">
                            <inertial mass="{mass(0.00899563)}" pos="2.96931e-12 0.0177547 0.00107314" quat="0.681301 0.732003 0 0"
                              diaginertia="{di(1.72352e-06, 1.60906e-06, 3.22006e-07)}"/>
                            <joint name="r2f85_right_driver_joint" class="r2f85_driver"/>
                            <geom mesh="r2f85_driver" material="r2f85_gray" class="r2f85_visual"/>
                            <geom mesh="r2f85_driver" class="r2f85_collision"/>
                            <body name="r2f85_right_coupler" pos="0 0.0315 -0.0041">
                              <inertial mass="{mass(0.0140974)}" pos="0 0.00301209 0.0232175" quat="0.705636 -0.0455904 0.0455904 0.705636"
                                diaginertia="{di(4.16206e-06, 3.52216e-06, 8.88131e-07)}"/>
                              <joint name="r2f85_right_coupler_joint" class="r2f85_coupler"/>
                              <geom mesh="r2f85_coupler" material="r2f85_black" class="r2f85_visual"/>
                              <geom mesh="r2f85_coupler" class="r2f85_collision"/>
                            </body>
                          </body>
                          <body name="r2f85_right_spring_link" pos="0 0.0132 0.0609">
                            <inertial mass="{mass(0.0221642)}" pos="-8.65005e-09 0.0181624 0.0212658" quat="0.663403 -0.244737 0.244737 0.663403"
                              diaginertia="{di(8.96853e-06, 6.71733e-06, 2.63931e-06)}"/>
                            <joint name="r2f85_right_spring_link_joint" class="r2f85_spring_link"/>
                            <geom mesh="r2f85_spring_link" material="r2f85_black" class="r2f85_visual"/>
                            <geom mesh="r2f85_spring_link" class="r2f85_collision"/>
                            <body name="r2f85_right_follower" pos="0 0.055 0.0375">
                              <inertial mass="{mass(0.0125222)}" pos="0 -0.011046 0.0124786" quat="1 0.1664 0 0"
                                diaginertia="{di(2.67415e-06, 2.4559e-06, 6.02031e-07)}"/>
                              <joint name="r2f85_right_follower_joint" class="r2f85_follower"/>
                              <geom mesh="r2f85_follower" material="r2f85_black" class="r2f85_visual"/>
                              <geom mesh="r2f85_follower" class="r2f85_collision"/>
                              <body name="r2f85_right_pad" pos="0 -0.0189 0.01352">
                                <inertial mass="{mass(0.0035)}" pos="0 -0.0025 0.0185" quat="0.707107 0 0 0.707107"
                                  diaginertia="{di(4.73958e-07, 3.64583e-07, 1.23958e-07)}"/>
                                <geom name="r2f85_right_pad1" type="box" mass="0" pos="0 -0.0026 0.028125"
                                  size="0.011 0.004 0.009375" friction="0.7" solimp="0.95 0.99 0.001"
                                  solref="0.004 1" priority="1" rgba="0.55 0.55 0.55 1"/>
                                <geom name="r2f85_right_pad2" type="box" mass="0" pos="0 -0.0026 0.009375"
                                  size="0.011 0.004 0.009375" friction="0.6" solimp="0.95 0.99 0.001"
                                  solref="0.004 1" priority="1" rgba="0.45 0.45 0.45 1"/>
                                <geom mesh="r2f85_pad" class="r2f85_visual"/>
                                <body name="r2f85_right_silicone_pad">
                                  <geom mesh="r2f85_silicone_pad" material="r2f85_black" class="r2f85_visual"/>
                                </body>
                              </body>
                            </body>
                          </body>

                          <!-- Left-hand side 4-bar linkage -->
                          <body name="r2f85_left_driver" pos="0 -0.0306011 0.054904" quat="0 0 0 1">
                            <inertial mass="{mass(0.00899563)}" pos="0 0.0177547 0.00107314" quat="0.681301 0.732003 0 0"
                              diaginertia="{di(1.72352e-06, 1.60906e-06, 3.22006e-07)}"/>
                            <joint name="r2f85_left_driver_joint" class="r2f85_driver"/>
                            <geom mesh="r2f85_driver" material="r2f85_gray" class="r2f85_visual"/>
                            <geom mesh="r2f85_driver" class="r2f85_collision"/>
                            <body name="r2f85_left_coupler" pos="0 0.0315 -0.0041">
                              <inertial mass="{mass(0.0140974)}" pos="0 0.00301209 0.0232175" quat="0.705636 -0.0455904 0.0455904 0.705636"
                                diaginertia="{di(4.16206e-06, 3.52216e-06, 8.88131e-07)}"/>
                              <joint name="r2f85_left_coupler_joint" class="r2f85_coupler"/>
                              <geom mesh="r2f85_coupler" material="r2f85_black" class="r2f85_visual"/>
                              <geom mesh="r2f85_coupler" class="r2f85_collision"/>
                            </body>
                          </body>
                          <body name="r2f85_left_spring_link" pos="0 -0.0132 0.0609" quat="0 0 0 1">
                            <inertial mass="{mass(0.0221642)}" pos="-8.65005e-09 0.0181624 0.0212658" quat="0.663403 -0.244737 0.244737 0.663403"
                              diaginertia="{di(8.96853e-06, 6.71733e-06, 2.63931e-06)}"/>
                            <joint name="r2f85_left_spring_link_joint" class="r2f85_spring_link"/>
                            <geom mesh="r2f85_spring_link" material="r2f85_black" class="r2f85_visual"/>
                            <geom mesh="r2f85_spring_link" class="r2f85_collision"/>
                            <body name="r2f85_left_follower" pos="0 0.055 0.0375">
                              <inertial mass="{mass(0.0125222)}" pos="0 -0.011046 0.0124786" quat="1 0.1664 0 0"
                                diaginertia="{di(2.67415e-06, 2.4559e-06, 6.02031e-07)}"/>
                              <joint name="r2f85_left_follower_joint" class="r2f85_follower"/>
                              <geom mesh="r2f85_follower" material="r2f85_black" class="r2f85_visual"/>
                              <geom mesh="r2f85_follower" class="r2f85_collision"/>
                              <body name="r2f85_left_pad" pos="0 -0.0189 0.01352">
                                <inertial mass="{mass(0.0035)}" pos="0 -0.0025 0.0185" quat="1 0 0 1"
                                  diaginertia="{di(4.73958e-07, 3.64583e-07, 1.23958e-07)}"/>
                                <geom name="r2f85_left_pad1" type="box" mass="0" pos="0 -0.0026 0.028125"
                                  size="0.011 0.004 0.009375" friction="0.7" solimp="0.95 0.99 0.001"
                                  solref="0.004 1" priority="1" rgba="0.55 0.55 0.55 1"/>
                                <geom name="r2f85_left_pad2" type="box" mass="0" pos="0 -0.0026 0.009375"
                                  size="0.011 0.004 0.009375" friction="0.6" solimp="0.95 0.99 0.001"
                                  solref="0.004 1" priority="1" rgba="0.45 0.45 0.45 1"/>
                                <geom mesh="r2f85_pad" class="r2f85_visual"/>
                                <body name="r2f85_left_silicone_pad">
                                  <geom mesh="r2f85_silicone_pad" material="r2f85_black" class="r2f85_visual"/>
                                </body>
                              </body>
                            </body>
                          </body>"""


def _robotiq_gripper_xml(spec: EndEffectorSpec, mount_override: MountOverride | None = None) -> str:
    """Robotiq 2F-85 adaptive gripper, ported from mujoco_menagerie's robotiq_2f85/2f85.xml.

    Default (mount_override is None or skip_base_mount is False): re-parented
    under the wrist flange at spec.mount_pos/mount_quat via r2f85_base_mount,
    exactly as upstream (today's Franka-relative behavior unchanged).

    When mount_override.skip_base_mount is set (Kinova's case, per the
    upstream kinova_gen3/README.md): r2f85_base_mount is omitted entirely --
    it duplicates Kinova's own end-effector interface -- and r2f85_base
    mounts directly at the override's mount_pos/mount_quat, with ee_site
    emitted as a sibling at the override's own ee_site_pos/ee_site_quat
    rather than nested inside r2f85_base.
    """
    children = _robotiq_base_children_xml(spec)

    if mount_override is None or not mount_override.skip_base_mount:
        return f"""                      <body name="r2f85_base_mount" pos="{_fmt(*spec.mount_pos)}" quat="{_fmt(*spec.mount_quat)}">
                        <geom mesh="r2f85_base_mount" material="r2f85_black" class="r2f85_visual"/>
                        <geom mesh="r2f85_base_mount" class="r2f85_collision"/>

                        <!-- End-effector site: same convention as Franka's ee_site (direct
                             child of the hand-equivalent body, so IK orientation tracking
                             is unaffected by which gripper is mounted) -->
                        <site name="ee_site" pos="{_fmt(*spec.ee_site_pos)}" size="0.01" rgba="0 1 0 0.5"/>

                        <body name="r2f85_base" pos="0 0 0.0038" quat="1 0 0 -1">
                          {children}
                        </body>
                      </body>"""

    return f"""                      <site name="ee_site" pos="{_fmt(*mount_override.ee_site_pos)}" quat="{_fmt(*mount_override.ee_site_quat)}" size="0.01" rgba="0 1 0 0.5"/>
                      <body name="r2f85_base" pos="{_fmt(*mount_override.mount_pos)}" quat="{_fmt(*mount_override.mount_quat)}">
                        {children}
                      </body>"""


def _gripper_xml(spec: EndEffectorSpec, mount_override: MountOverride | None = None) -> str:
    if spec.name == "franka":
        return _franka_gripper_xml(spec)
    return _robotiq_gripper_xml(spec, mount_override)


def _ee_default_block(spec: EndEffectorSpec) -> str:
    if spec.name != "robotiq_2f85":
        return ""
    return """

    <!-- Robotiq 2F-85 defaults (alternative end-effector, see controllers/end_effectors.py) -->
    <default class="r2f85">
      <joint axis="1 0 0"/>
      <default class="r2f85_driver">
        <joint range="0 0.8" armature="0.005" damping="0.1"
               solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
      </default>
      <default class="r2f85_follower">
        <joint pos="0 -0.018 0.0065" range="-0.872664 0.872664" armature="0.001"
               solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
      </default>
      <default class="r2f85_spring_link">
        <joint range="-0.29670597283 0.8" armature="0.001"
               stiffness="0.05" springref="2.62" damping="0.00125"/>
      </default>
      <default class="r2f85_coupler">
        <joint range="-1.57 0" armature="0.001"
               solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
      </default>
      <default class="r2f85_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="r2f85_collision">
        <geom type="mesh" group="3"/>
      </default>
    </default>"""


def _ee_asset_block(spec: EndEffectorSpec) -> str:
    if spec.name != "robotiq_2f85":
        return ""
    return """

    <!-- Robotiq 2F-85 materials (prefixed to avoid conflicts) -->
    <material name="r2f85_black" rgba="0.149 0.149 0.149 1"/>
    <material name="r2f85_gray"  rgba="0.4627 0.4627 0.4627 1"/>

    <!-- Robotiq 2F-85 meshes -->
    <mesh name="r2f85_base_mount"   file="../assets/robotiq_2f85/assets/base_mount.stl"   scale="0.001 0.001 0.001"/>
    <mesh name="r2f85_base"         file="../assets/robotiq_2f85/assets/base.stl"         scale="0.001 0.001 0.001"/>
    <mesh name="r2f85_driver"       file="../assets/robotiq_2f85/assets/driver.stl"       scale="0.001 0.001 0.001"/>
    <mesh name="r2f85_coupler"      file="../assets/robotiq_2f85/assets/coupler.stl"      scale="0.001 0.001 0.001"/>
    <mesh name="r2f85_follower"     file="../assets/robotiq_2f85/assets/follower.stl"     scale="0.001 0.001 0.001"/>
    <mesh name="r2f85_pad"          file="../assets/robotiq_2f85/assets/pad.stl"          scale="0.001 0.001 0.001"/>
    <mesh name="r2f85_silicone_pad" file="../assets/robotiq_2f85/assets/silicone_pad.stl" scale="0.001 0.001 0.001"/>
    <mesh name="r2f85_spring_link"  file="../assets/robotiq_2f85/assets/spring_link.stl"  scale="0.001 0.001 0.001"/>"""


def _ee_actuator_xml(spec: EndEffectorSpec) -> str:
    if spec.name == "franka":
        return f"""    <general class="ee_actuator" name="{spec.actuator_name}" tendon="panda_split"
             forcerange="-100 100" ctrlrange="0 255"
             gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>"""
    return f"""    <general class="ee_actuator" name="{spec.actuator_name}" tendon="panda_split"
             forcerange="-5 5" ctrlrange="0 255"
             gainprm="0.3137255 0 0" biasprm="0 -100 -10"/>"""


def _ee_tendon_xml(spec: EndEffectorSpec) -> str:
    joints = "\n".join(f'      <joint joint="{j}" coef="0.5"/>' for j in spec.driver_joints)
    return f"""    <fixed name="panda_split">
{joints}
    </fixed>"""


def _ee_equality_xml(spec: EndEffectorSpec) -> str:
    if spec.name == "franka":
        return """    <joint joint1="finger_joint1" joint2="finger_joint2"
           solimp="0.95 0.99 0.001" solref="0.005 1"/>"""
    return """    <connect anchor="0 0 0" body1="r2f85_right_follower" body2="r2f85_right_coupler"
             solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <connect anchor="0 0 0" body1="r2f85_left_follower" body2="r2f85_left_coupler"
             solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <joint joint1="r2f85_right_driver_joint" joint2="r2f85_left_driver_joint"
           polycoef="0 1 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1"/>"""


def _ee_contact_exclude_xml(
    spec: EndEffectorSpec, arm_spec: ArmSpec, mount_override: MountOverride | None
) -> str:
    if spec.name == "franka":
        return f'    <exclude body1="{arm_spec.wrist_body}" body2="panda_hand"/>'
    base_attach_body = (
        "r2f85_base" if mount_override and mount_override.skip_base_mount else "r2f85_base_mount"
    )
    return f"""    <exclude body1="{arm_spec.wrist_body}" body2="{base_attach_body}"/>
    <exclude body1="r2f85_base" body2="r2f85_left_driver"/>
    <exclude body1="r2f85_base" body2="r2f85_right_driver"/>
    <exclude body1="r2f85_base" body2="r2f85_left_spring_link"/>
    <exclude body1="r2f85_base" body2="r2f85_right_spring_link"/>
    <exclude body1="r2f85_right_coupler" body2="r2f85_right_follower"/>
    <exclude body1="r2f85_left_coupler" body2="r2f85_left_follower"/>"""


def _ee_keyframe(spec: EndEffectorSpec, arm_spec: ArmSpec) -> tuple[str, str, str]:
    """Return (qpos_tail, ctrl_tail, qpos_layout_comment) for the home keyframe."""
    ctrl_tail = f"{spec.open_ctrl:.6g}"
    if spec.name == "franka":
        qpos_tail = "0.04 0.04"
        comment = f"""qpos layout (35 total):
        [0:3]   base_link position (x, y, z)
        [3:7]   base_link quaternion (w, x, y, z)
        [7:19]  leg joints FR/FL/RR/RL x [hip, thigh, calf]
        [19:26] {arm_spec.display_name} joint1-7
        [26:28] finger_joint1, finger_joint2
        [28:35] target_cube freejoint (x, y, z, qw, qx, qy, qz)
      ctrl layout (20 total):
        [0:12]  Go2 leg motors (FR, FL, RR, RL order)
        [12:19] {arm_spec.display_name} arm actuators 1-7
        [19]    Gripper actuator8"""
    else:
        qpos_tail = "0 0 0 0 0 0 0 0"
        comment = f"""qpos layout (41 total):
        [0:3]   base_link position (x, y, z)
        [3:7]   base_link quaternion (w, x, y, z)
        [7:19]  leg joints FR/FL/RR/RL x [hip, thigh, calf]
        [19:26] {arm_spec.display_name} joint1-7
        [26:34] Robotiq 2F-85: right_driver, right_coupler, right_spring_link,
                right_follower, left_driver, left_coupler, left_spring_link, left_follower
        [34:41] target_cube freejoint (x, y, z, qw, qx, qy, qz)
      ctrl layout (20 total):
        [0:12]  Go2 leg motors (FR, FL, RR, RL order)
        [12:19] {arm_spec.display_name} arm actuators 1-7
        [19]    Gripper actuator8 (Robotiq fingers_actuator-equivalent)"""
    return qpos_tail, ctrl_tail, comment


# ─────────────────────────────────────────────────────────────────────────
# Arm defaults / assets / body chain / actuators (arm dimension)
# ─────────────────────────────────────────────────────────────────────────

def _franka_arm_default_block() -> str:
    """Franka Panda defaults -- identical to the original literal block."""
    return """

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
    </default>"""


def _kinova_arm_default_block() -> str:
    """Kinova Gen3 defaults, copied verbatim from upstream gen3.xml's
    visual/collision/large_actuator/small_actuator classes (kinova_-prefixed).
    Upstream has no joint-level defaults at all (no armature/damping
    specified) -- intentionally not inventing one; joints fall back to
    MuJoCo's compiled-in defaults, matching upstream behavior exactly.
    """
    return """

    <!-- Kinova Gen3 defaults (alternative arm, see controllers/arms.py) -->
    <default class="kinova">
      <default class="kinova_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.75294 0.75294 0.75294 1"/>
      </default>
      <default class="kinova_collision">
        <geom type="mesh" group="3"/>
      </default>
      <default class="kinova_large_actuator">
        <position kp="2000" kv="100" forcerange="-105 105"/>
      </default>
      <default class="kinova_small_actuator">
        <position kp="500" kv="50" forcerange="-52 52"/>
      </default>
    </default>"""


def _arm_default_block(arm_spec: ArmSpec) -> str:
    if arm_spec.actuator_kind == "general_pd":
        return _franka_arm_default_block()
    return _kinova_arm_default_block()


def _franka_arm_asset_block() -> str:
    """Panda collision + visual meshes -- identical to the original literal block."""
    return """

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
    <mesh name="p_finger_1" file="../assets/panda/assets/finger_1.obj"/>"""


def _kinova_arm_asset_block() -> str:
    """Kinova Gen3 meshes, kinova_-prefixed, no scale attribute (already in meters)."""
    return """

    <!-- Kinova Gen3 meshes -->
    <mesh name="kinova_base_link"              file="../assets/kinova_gen3/assets/base_link.stl"/>
    <mesh name="kinova_shoulder_link"          file="../assets/kinova_gen3/assets/shoulder_link.stl"/>
    <mesh name="kinova_half_arm_1_link"        file="../assets/kinova_gen3/assets/half_arm_1_link.stl"/>
    <mesh name="kinova_half_arm_2_link"        file="../assets/kinova_gen3/assets/half_arm_2_link.stl"/>
    <mesh name="kinova_forearm_link"           file="../assets/kinova_gen3/assets/forearm_link.stl"/>
    <mesh name="kinova_spherical_wrist_1_link" file="../assets/kinova_gen3/assets/spherical_wrist_1_link.stl"/>
    <mesh name="kinova_spherical_wrist_2_link" file="../assets/kinova_gen3/assets/spherical_wrist_2_link.stl"/>
    <mesh name="kinova_bracelet_link"          file="../assets/kinova_gen3/assets/bracelet_with_vision_link.stl"/>"""


def _arm_asset_block(arm_spec: ArmSpec) -> str:
    if arm_spec.actuator_kind == "general_pd":
        return _franka_arm_asset_block()
    return _kinova_arm_asset_block()


def _franka_arm_body_xml(
    arm_spec: ArmSpec, ee_spec: EndEffectorSpec, mount_override: MountOverride | None
) -> str:
    """Franka panda_link0..panda_link7 chain -- identical to the original
    literal block, parameterized on arm_spec.joint_names."""
    j = arm_spec.joint_names
    return f"""      <body name="panda_link0" pos="0 0 0.10" childclass="panda">
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
          <joint name="{j[0]}" class="panda"/>
          <geom material="white" mesh="p_link1" class="panda_visual"/>
          <geom mesh="p_link1_c" class="panda_collision"/>

          <body name="panda_link2" quat="1 -1 0 0">
            <inertial mass="0.226424" pos="-0.003141 -0.02872 0.003495"
              fullinertia="0.002787 0.009839 0.009099 -0.001374 0.003589 2.464e-4"/>
            <joint name="{j[1]}" class="panda" range="-1.7628 1.7628"/>
            <geom material="white" mesh="p_link2" class="panda_visual"/>
            <geom mesh="p_link2_c" class="panda_collision"/>

            <body name="panda_link3" pos="0 -0.316 0" quat="1 1 0 0">
              <inertial mass="1.130011" pos="2.7518e-2 3.9252e-2 -6.6502e-2"
                fullinertia="0.013035 0.012654 0.003791 -0.001666 -0.003989 -0.004482"/>
              <joint name="{j[2]}" class="panda"/>
              <geom mesh="p_link3_0" material="white"      class="panda_visual"/>
              <geom mesh="p_link3_1" material="white"      class="panda_visual"/>
              <geom mesh="p_link3_2" material="white"      class="panda_visual"/>
              <geom mesh="p_link3_3" material="p_dark"    class="panda_visual"/>
              <geom mesh="p_link3_c" class="panda_collision"/>

              <body name="panda_link4" pos="0.0825 0 0" quat="1 1 0 0">
                <inertial mass="1.255763" pos="-5.317e-2 1.04419e-1 2.7454e-2"
                  fullinertia="0.009049 0.006843 0.009913 0.002729 -4.662e-4 0.003024"/>
                <joint name="{j[3]}" class="panda" range="-3.0718 -0.0698"/>
                <geom mesh="p_link4_0" material="white"      class="panda_visual"/>
                <geom mesh="p_link4_1" material="white"      class="panda_visual"/>
                <geom mesh="p_link4_2" material="p_dark"    class="panda_visual"/>
                <geom mesh="p_link4_3" material="white"      class="panda_visual"/>
                <geom mesh="p_link4_c" class="panda_collision"/>

                <body name="panda_link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                  <inertial mass="0.429081" pos="-1.1953e-2 4.1065e-2 -3.8437e-2"
                    fullinertia="0.012442 0.010316 0.003019 -7.410e-4 -0.001413 8.015e-5"/>
                  <joint name="{j[4]}" class="panda"/>
                  <geom mesh="p_link5_0" material="p_dark"        class="panda_visual"/>
                  <geom mesh="p_link5_1" material="white"          class="panda_visual"/>
                  <geom mesh="p_link5_2" material="white"          class="panda_visual"/>
                  <geom mesh="p_link5_c0" class="panda_collision"/>
                  <geom mesh="p_link5_c1" class="panda_collision"/>
                  <geom mesh="p_link5_c2" class="panda_collision"/>

                  <body name="panda_link6" quat="1 1 0 0">
                    <inertial mass="0.583294" pos="6.0149e-2 -1.4117e-2 -1.0517e-2"
                      fullinertia="6.874e-4 1.524e-3 1.902e-3 3.815e-5 -4.053e-4 1.194e-4"/>
                    <joint name="{j[5]}" class="panda" range="-0.0175 3.7525"/>
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
                      <joint name="{j[6]}" class="panda"/>
                      <geom mesh="p_link7_0" material="white"  class="panda_visual"/>
                      <geom mesh="p_link7_1" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_2" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_3" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_4" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_5" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_6" material="p_dark" class="panda_visual"/>
                      <geom mesh="p_link7_7" material="white"  class="panda_visual"/>
                      <geom mesh="p_link7_c" class="panda_collision"/>

{_gripper_xml(ee_spec, mount_override)}
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>"""


def _kinova_arm_body_xml(
    arm_spec: ArmSpec, ee_spec: EndEffectorSpec, mount_override: MountOverride | None
) -> str:
    """Kinova Gen3 base_link..bracelet_link chain, ported verbatim from
    mujoco_menagerie's kinova_gen3/gen3.xml. Masses/diaginertia scaled by
    arm_spec.mass_scale (currently 1.0, a numeric no-op, but kept uniform
    with the Robotiq scaling path). Joints named via arm_spec.joint_names;
    only joints 2/4/6 (indices 1,3,5) carry an explicit hardware range --
    joints 1/3/5/7 are continuous on real hardware and omit range entirely
    (compiler autolimits="true" makes that unlimited).
    """
    j = arm_spec.joint_names
    s = arm_spec.mass_scale

    def mass(value: float) -> str:
        return f"{value * s:.6g}"

    def di(*values: float) -> str:
        return _scale_inertia_by(" ".join(str(v) for v in values), s)

    return f"""      <body name="kinova_base_link" pos="0 0 0.10" childclass="kinova">
        <inertial mass="{mass(1.697)}" pos="-0.000648 -0.000166 0.084487"
          quat="0.999294 0.00139618 -0.0118387 0.035636"
          diaginertia="{di(0.00462407, 0.00449437, 0.00207755)}"/>
        <geom mesh="kinova_base_link" class="kinova_visual"/>
        <geom mesh="kinova_base_link" class="kinova_collision"/>

        <body name="kinova_shoulder_link" pos="0 0 0.15643" quat="0 1 0 0">
          <inertial mass="{mass(1.3773)}" pos="-2.3e-05 -0.010364 -0.07336"
            quat="0.707051 0.0451246 -0.0453544 0.704263"
            diaginertia="{di(0.00488868, 0.00457, 0.00135132)}"/>
          <joint name="{j[0]}"/>
          <geom mesh="kinova_shoulder_link" class="kinova_visual"/>
          <geom mesh="kinova_shoulder_link" class="kinova_collision"/>

          <body name="kinova_half_arm_1_link" pos="0 0.005375 -0.12838" quat="1 1 0 0">
            <inertial mass="{mass(1.1636)}" pos="-4.4e-05 -0.09958 -0.013278"
              quat="0.482348 0.516286 -0.516862 0.483366"
              diaginertia="{di(0.0113017, 0.011088, 0.00102532)}"/>
            <joint name="{j[1]}" range="-2.24 2.24"/>
            <geom mesh="kinova_half_arm_1_link" class="kinova_visual"/>
            <geom mesh="kinova_half_arm_1_link" class="kinova_collision"/>

            <body name="kinova_half_arm_2_link" pos="0 -0.21038 -0.006375" quat="1 -1 0 0">
              <inertial mass="{mass(1.1636)}" pos="-4.4e-05 -0.006641 -0.117892"
                quat="0.706144 0.0213722 -0.0209128 0.707437"
                diaginertia="{di(0.0111633, 0.010932, 0.00100671)}"/>
              <joint name="{j[2]}"/>
              <geom mesh="kinova_half_arm_2_link" class="kinova_visual"/>
              <geom mesh="kinova_half_arm_2_link" class="kinova_collision"/>

              <body name="kinova_forearm_link" pos="0 0.006375 -0.21038" quat="1 1 0 0">
                <inertial mass="{mass(0.9302)}" pos="-1.8e-05 -0.075478 -0.015006"
                  quat="0.483678 0.515961 -0.515859 0.483455"
                  diaginertia="{di(0.00834839, 0.008147, 0.000598606)}"/>
                <joint name="{j[3]}" range="-2.57 2.57"/>
                <geom mesh="kinova_forearm_link" class="kinova_visual"/>
                <geom mesh="kinova_forearm_link" class="kinova_collision"/>

                <body name="kinova_spherical_wrist_1_link" pos="0 -0.20843 -0.006375" quat="1 -1 0 0">
                  <inertial mass="{mass(0.6781)}" pos="1e-06 -0.009432 -0.063883"
                    quat="0.703558 0.0707492 -0.0707492 0.703558"
                    diaginertia="{di(0.00165901, 0.001596, 0.000346988)}"/>
                  <joint name="{j[4]}"/>
                  <geom mesh="kinova_spherical_wrist_1_link" class="kinova_visual"/>
                  <geom mesh="kinova_spherical_wrist_1_link" class="kinova_collision"/>

                  <body name="kinova_spherical_wrist_2_link" pos="0 0.00017505 -0.10593" quat="1 1 0 0">
                    <inertial mass="{mass(0.6781)}" pos="1e-06 -0.045483 -0.00965"
                      quat="0.44426 0.550121 -0.550121 0.44426"
                      diaginertia="{di(0.00170087, 0.001641, 0.00035013)}"/>
                    <joint name="{j[5]}" range="-2.09 2.09"/>
                    <geom mesh="kinova_spherical_wrist_2_link" class="kinova_visual"/>
                    <geom mesh="kinova_spherical_wrist_2_link" class="kinova_collision"/>

                    <body name="kinova_bracelet_link" pos="0 -0.10593 -0.00017505" quat="1 -1 0 0">
                      <inertial mass="{mass(0.5)}" pos="0.000281 0.011402 -0.029798"
                        quat="0.394358 0.596779 -0.577293 0.393789"
                        diaginertia="{di(0.000657336, 0.000587019, 0.000320645)}"/>
                      <joint name="{j[6]}"/>
                      <geom mesh="kinova_bracelet_link" class="kinova_visual"/>
                      <geom mesh="kinova_bracelet_link" class="kinova_collision"/>

{_gripper_xml(ee_spec, mount_override)}
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>"""


def _arm_body_xml(
    arm_spec: ArmSpec, ee_spec: EndEffectorSpec, mount_override: MountOverride | None
) -> str:
    if arm_spec.actuator_kind == "general_pd":
        return _franka_arm_body_xml(arm_spec, ee_spec, mount_override)
    return _kinova_arm_body_xml(arm_spec, ee_spec, mount_override)


def _franka_arm_actuator_xml(arm_spec: ArmSpec) -> str:
    j, a = arm_spec.joint_names, arm_spec.actuator_names
    return f"""    <general class="panda" name="{a[0]}" joint="{j[0]}"
             gainprm="4500" biasprm="0 -4500 -450"/>
    <general class="panda" name="{a[1]}" joint="{j[1]}"
             gainprm="4500" biasprm="0 -4500 -450" ctrlrange="-1.7628 1.7628"/>
    <general class="panda" name="{a[2]}" joint="{j[2]}"
             gainprm="3500" biasprm="0 -3500 -350"/>
    <general class="panda" name="{a[3]}" joint="{j[3]}"
             gainprm="3500" biasprm="0 -3500 -350" ctrlrange="-3.0718 -0.0698"/>
    <general class="panda" name="{a[4]}" joint="{j[4]}"
             gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <general class="panda" name="{a[5]}" joint="{j[5]}"
             gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"
             ctrlrange="-0.0175 3.7525"/>
    <general class="panda" name="{a[6]}" joint="{j[6]}"
             gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>"""


def _kinova_arm_actuator_xml(arm_spec: ArmSpec) -> str:
    j, a = arm_spec.joint_names, arm_spec.actuator_names
    return f"""    <position class="kinova_large_actuator" name="{a[0]}" joint="{j[0]}"/>
    <position class="kinova_large_actuator" name="{a[1]}" joint="{j[1]}"
              ctrlrange="-2.2497294058206907 2.2497294058206907"/>
    <position class="kinova_large_actuator" name="{a[2]}" joint="{j[2]}"/>
    <position class="kinova_large_actuator" name="{a[3]}" joint="{j[3]}"
              ctrlrange="-2.5795966344476193 2.5795966344476193"/>
    <position class="kinova_small_actuator" name="{a[4]}" joint="{j[4]}"/>
    <position class="kinova_small_actuator" name="{a[5]}" joint="{j[5]}"
              ctrlrange="-2.0996310901491784 2.0996310901491784"/>
    <position class="kinova_small_actuator" name="{a[6]}" joint="{j[6]}"/>"""


def _arm_actuator_xml(arm_spec: ArmSpec) -> str:
    if arm_spec.actuator_kind == "general_pd":
        return _franka_arm_actuator_xml(arm_spec)
    return _kinova_arm_actuator_xml(arm_spec)


_ARM_BODY_CHAIN: dict[str, tuple[str, ...]] = {
    "franka": (
        "panda_link0", "panda_link1", "panda_link2", "panda_link3",
        "panda_link4", "panda_link5", "panda_link6", "panda_link7",
    ),
    "kinova_gen3": (
        "kinova_base_link", "kinova_shoulder_link", "kinova_half_arm_1_link",
        "kinova_half_arm_2_link", "kinova_forearm_link",
        "kinova_spherical_wrist_1_link", "kinova_spherical_wrist_2_link",
        "kinova_bracelet_link",
    ),
}


def _arm_contact_exclude_xml(arm_spec: ArmSpec) -> str:
    chain = _ARM_BODY_CHAIN[arm_spec.name]
    return "\n".join(
        f'    <exclude body1="{a}" body2="{b}"/>' for a, b in zip(chain, chain[1:])
    )


def _workspace_camera_xml() -> str:
    """Fixed, world-mounted camera framing the cube's full reachable region
    (x in [1.36, 1.84], y in [-0.33, 0.33] -- see README's --cube-pos note)
    throughout WALKING and APPROACHING. Placed beyond the table's far edge,
    elevated, angled back across the table so the approaching Go2 body never
    occludes the cube. Not arm-mounted -- see perception/ package (added in
    a later phase of this feature)."""
    return (
        '    <camera name="workspace_cam" pos="2.35 0 0.95" '
        'xyaxes="0 1 0 -0.55 0 0.83"/>'
    )


def build_combined_xml(
    arm: str = DEFAULT_ARM,
    end_effector: str | None = None,
    scene: str = "default",
) -> str:
    """Return the complete combined MJCF XML as a string."""
    arm_spec = get_arm_spec(arm)
    ee_name = end_effector or arm_spec.default_end_effector
    validate_combo(arm_spec.name, ee_name)
    spec = get_spec(ee_name)
    mount_override = get_mount_override(arm_spec.name, spec.name)
    qpos_tail, ctrl_tail, keyframe_comment = _ee_keyframe(spec, arm_spec)
    home = _fmt(*arm_spec.home_pose)

    if scene not in SCENE_PRESETS:
        raise ValueError(
            f"Unknown scene {scene!r}. Valid options: {sorted(SCENE_PRESETS)}"
        )
    sc = SCENE_PRESETS[scene]

    xml = f"""<mujoco model="go2_{arm_spec.name}">
  <!-- ARM_STAMP: {arm_spec.name} -->
  <!-- END_EFFECTOR_STAMP: {spec.name} -->
  <!-- SCENE_STAMP: {scene} -->
  <!--
    Combined Unitree Go2 + {arm_spec.display_name} loco-manipulation model.
    Coordinate frame: X-forward, Y-left, Z-up (standard robotics).
    {arm_spec.display_name} root ({arm_spec.root_body}) is rigidly mounted at pos="0 0 0.10" on Go2 base_link.
    {arm_spec.display_name} masses scaled by {arm_spec.mass_scale} (relative to upstream).
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
    <headlight ambient="{sc['headlight_ambient']}" diffuse="{sc['headlight_diffuse']}" specular="0.1 0.1 0.1"/>
    <map shadowclip="2.0" shadowscale="{sc['shadowscale']}" fogstart="{sc['fogstart']}" fogend="{sc['fogend']}"/>
    <rgba haze="{sc['haze']}"/>
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

    <!-- End-effector actuator default (arm-agnostic, so the gripper actuator
         doesn't depend on whichever arm-specific default class is active) -->
    <default class="ee_actuator">
      <general dyntype="none" biastype="affine"/>
    </default>
{_arm_default_block(arm_spec)}
{_ee_default_block(spec)}
  </default>

  <!-- ── Assets ──────────────────────────────────────────────────────── -->
  <asset>
    <!-- Skybox: gradient set by --scene preset -->
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="{sc['skybox_rgb1']}" rgb2="{sc['skybox_rgb2']}" width="512" height="512"/>

    <!-- Floor: checker tile, colours set by --scene preset -->
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="{sc['floor_rgb1']}" rgb2="{sc['floor_rgb2']}"
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

    <!-- Materials: Panda (prefixed to avoid conflict with Go2 names; declared
         unconditionally -- harmless if unused when a non-Franka arm is built) -->
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
{_arm_asset_block(arm_spec)}
{_ee_asset_block(spec)}
  </asset>

  <!-- ── World ───────────────────────────────────────────────────────── -->
  <worldbody>
    <!-- Key light: position/colour set by --scene preset, casts shadows -->
    <light name="sun" directional="true" pos="{sc['sun_pos']}" dir="{sc['sun_dir']}"
           diffuse="{sc['sun_diffuse']}" specular="{sc['sun_specular']}" castshadow="true"/>
    <!-- Fill light: soft fill, no shadow -->
    <light name="fill" directional="true" pos="{sc['fill_pos']}" dir="{sc['fill_dir']}"
           diffuse="{sc['fill_diffuse']}" specular="0.02 0.02 0.02" castshadow="false"/>
    <!-- Rim light: edge separation, no shadow -->
    <light name="rim" directional="true" pos="{sc['rim_pos']}" dir="{sc['rim_dir']}"
           diffuse="{sc['rim_diffuse']}" specular="0.0 0.0 0.0" castshadow="false"/>

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

      <!-- ── {arm_spec.display_name} arm (rigidly mounted on trunk) ───── -->
      <!--
        Mounting geometry:
          base_link origin: robot body center (~0.445 m above ground at rest)
          Trunk box half-size Z: 0.057 m → trunk top at +0.057 m from base_link
          Mount offset: 0.10 m above base_link (safely above trunk top)
          No joint → rigid attachment
      -->
{_arm_body_xml(arm_spec, spec, mount_override)}

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

{_workspace_camera_xml()}

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

    <!-- Arm: position control (Franka: integrated PD via general; Kinova: native position servo) -->
{_arm_actuator_xml(arm_spec)}
    <!-- Gripper: tendon-based, ctrl range 0-255 -->
{_ee_actuator_xml(spec)}
  </actuator>

  <!-- ── Tendons ──────────────────────────────────────────────────────── -->
  <tendon>
    <!-- Finger synchronization: both fingers move together -->
{_ee_tendon_xml(spec)}
  </tendon>

  <!-- ── Equality constraints ─────────────────────────────────────────── -->
  <equality>
    <!-- Finger mimic / 4-bar linkage coupling -->
{_ee_equality_xml(spec)}
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

    <!-- Arm end-effector tracking -->
    <framepos  name="ee_pos"  objtype="site" objname="ee_site"/>
    <framequat name="ee_quat" objtype="site" objname="ee_site"/>

    <!-- Cube position tracking -->
    <framepos name="cube_pos" objtype="site" objname="cube_center"/>
  </sensor>

  <!-- ── Contact exclusions ───────────────────────────────────────────── -->
  <contact>
    <!-- Arm adjacent-link exclusions (prevent self-collision artifacts) -->
{_arm_contact_exclude_xml(arm_spec)}
{_ee_contact_exclude_xml(spec, arm_spec, mount_override)}
    <!-- Arm-Go2 trunk exclusion (mounting interface) -->
    <exclude body1="base_link"   body2="{arm_spec.root_body}"/>
  </contact>

  <!-- ── Keyframe ─────────────────────────────────────────────────────── -->
  <keyframe>
    <!--
      {keyframe_comment}
    -->
    <key name="home"
      qpos="0 0 0.27 1 0 0 0
            0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8
            {home}
            {qpos_tail}
            1.6 0 0.325 1 0 0 0"
      ctrl="0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8
            {home}  {ctrl_tail}"/>
  </keyframe>

</mujoco>
"""
    return xml


def main(
    arm: str = DEFAULT_ARM,
    end_effector: str | None = None,
    scene: str = "default",
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "combined.xml"
    out_path.write_text(build_combined_xml(arm, end_effector, scene), encoding="utf-8")
    arm_spec = get_arm_spec(arm)
    ee_name = end_effector or arm_spec.default_end_effector
    print(f"Written: {out_path}  (arm={arm}, end_effector={ee_name}, scene={scene})")
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
    import argparse

    _SCENES = sorted(SCENE_PRESETS)
    p = argparse.ArgumentParser(description="Build the combined Go2+arm MJCF model.")
    p.add_argument(
        "--arm", choices=sorted(ARMS), default=DEFAULT_ARM,
        help=f"Arm to mount (default: {DEFAULT_ARM})",
    )
    p.add_argument(
        "--end-effector", dest="end_effector", default=None,
        help="End-effector override (default: arm's own default)",
    )
    p.add_argument(
        "--scene", choices=_SCENES, default="default",
        help=f"Visual scene preset: {_SCENES} (default: default)",
    )
    args = p.parse_args()
    main(arm=args.arm, end_effector=args.end_effector, scene=args.scene)
