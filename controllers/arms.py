"""
Single source of truth for per-arm metadata.

Consumed by:
  - scripts/build_model.py        (MJCF body chain / default / actuator generation)
  - controllers/manipulation.py   (joint/actuator id lookups, home pose, IK limits)
  - controllers/coordinator.py    (threads `arm` through to ManipulationController)

Adding a new arm means adding one ArmSpec entry to ARMS plus a corresponding
_<name>_arm_body_xml() / default / actuator helper in build_model.py. No other
file should need editing to support the variant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ArmSpec:
    name: str
    display_name: str

    # Body chain anchors
    root_body: str   # mounted on Go2's base_link
    wrist_body: str  # gripper mount parent (flange frame)

    # 7-DOF joint/actuator names, in order
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]

    # Home keyframe (rad), 7 values matching joint_names order
    home_pose: tuple[float, ...]

    # Joint limits (rad); unranged real joints use +/-math.inf
    q_lo: tuple[float, ...]
    q_hi: tuple[float, ...]

    # Inertia scale factor applied to every body in the arm chain
    mass_scale: float

    # Drives which actuator XML shape build_model.py emits for this arm
    actuator_kind: str  # "general_pd" | "position_pd"

    # End-effector compatibility
    default_end_effector: str
    allowed_end_effectors: frozenset[str]


FRANKA = ArmSpec(
    name="franka",
    display_name="Franka Panda",
    root_body="panda_link0",
    wrist_body="panda_link7",
    joint_names=(
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7",
    ),
    actuator_names=(
        "actuator1", "actuator2", "actuator3", "actuator4",
        "actuator5", "actuator6", "actuator7",
    ),
    home_pose=(0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853),
    q_lo=(-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
    q_hi=(2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
    mass_scale=0.35,
    actuator_kind="general_pd",
    default_end_effector="franka",
    allowed_end_effectors=frozenset({"franka", "robotiq_2f85"}),
)

KINOVA_GEN3 = ArmSpec(
    name="kinova_gen3",
    display_name="Kinova Gen3",
    root_body="kinova_base_link",
    wrist_body="kinova_bracelet_link",
    joint_names=(
        "kinova_joint_1", "kinova_joint_2", "kinova_joint_3", "kinova_joint_4",
        "kinova_joint_5", "kinova_joint_6", "kinova_joint_7",
    ),
    actuator_names=(
        "kinova_actuator_1", "kinova_actuator_2", "kinova_actuator_3", "kinova_actuator_4",
        "kinova_actuator_5", "kinova_actuator_6", "kinova_actuator_7",
    ),
    # Upstream "home" keyframe (gen3.xml)
    home_pose=(0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633),
    # Only joints 2/4/6 have finite hardware ranges; 1/3/5/7 are continuous.
    q_lo=(-math.inf, -2.24, -math.inf, -2.57, -math.inf, -2.09, -math.inf),
    q_hi=(math.inf, 2.24, math.inf, 2.57, math.inf, 2.09, math.inf),
    mass_scale=1.0,
    actuator_kind="position_pd",
    default_end_effector="robotiq_2f85",
    allowed_end_effectors=frozenset({"robotiq_2f85"}),
)

ARMS: dict[str, ArmSpec] = {
    FRANKA.name: FRANKA,
    KINOVA_GEN3.name: KINOVA_GEN3,
}

DEFAULT_ARM = FRANKA.name


def get_arm_spec(name: str) -> ArmSpec:
    try:
        return ARMS[name]
    except KeyError:
        valid = ", ".join(sorted(ARMS))
        raise ValueError(f"Unknown arm {name!r}. Valid options: {valid}") from None


def validate_combo(arm_name: str, ee_name: str) -> None:
    spec = get_arm_spec(arm_name)
    if ee_name not in spec.allowed_end_effectors:
        valid = ", ".join(sorted(spec.allowed_end_effectors))
        raise ValueError(
            f"End-effector {ee_name!r} is not compatible with arm {arm_name!r}. "
            f"Valid options for {arm_name!r}: {valid}"
        )
