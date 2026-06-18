"""
Single source of truth for per-end-effector metadata.

Consumed by:
  - scripts/build_model.py        (MJCF subtree / actuator / tendon generation)
  - controllers/manipulation.py   (actuator + finger body id lookups, ctrl polarity)
  - controllers/coordinator.py    (fingertip-to-ee_site waypoint offset)

Adding a new end-effector means adding one EndEffectorSpec entry to
END_EFFECTORS plus a corresponding _build_<name>_gripper_xml() helper in
build_model.py. No other file should need editing to support the variant.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EndEffectorSpec:
    name: str
    display_name: str

    # MJCF mounting (relative to panda_link7 -- the wrist flange frame)
    mount_pos: tuple[float, float, float]
    mount_quat: tuple[float, float, float, float]
    ee_site_pos: tuple[float, float, float]

    # Actuation
    actuator_name: str
    open_ctrl: float
    close_ctrl: float

    # Contact-based grasp detection (is_grasped())
    left_finger_body: str
    right_finger_body: str

    # Vertical offset (m) from ee_site to the fingertip/pad contact plane,
    # used by TaskCoordinator._compute_waypoints() for descend/lower targets.
    ftp_offset: float

    # Inertia scale factor applied to every body in the gripper subtree.
    mass_scale: float

    # Actuated joint names coupled by the gripper's fixed tendon.
    driver_joints: tuple[str, ...] = field(default_factory=tuple)


FRANKA_HAND = EndEffectorSpec(
    name="franka",
    display_name="Franka Panda stock two-finger gripper",
    mount_pos=(0.0, 0.0, 0.107),
    mount_quat=(0.9238795, 0.0, 0.0, -0.3826834),
    ee_site_pos=(0.0, 0.0, 0.12),
    actuator_name="actuator8",
    open_ctrl=255.0,
    close_ctrl=0.0,
    left_finger_body="panda_left_finger",
    right_finger_body="panda_right_finger",
    ftp_offset=0.015,
    mass_scale=0.35,
    driver_joints=("finger_joint1", "finger_joint2"),
)

ROBOTIQ_2F85 = EndEffectorSpec(
    name="robotiq_2f85",
    display_name="Robotiq 2F-85 adaptive gripper",
    mount_pos=(0.0, 0.0, 0.107),
    mount_quat=(0.9238795, 0.0, 0.0, -0.3826834),
    ee_site_pos=(0.0, 0.0, 0.1488),
    actuator_name="actuator8",
    # Verified empirically: driver_joint range is [0, 0.8] starting at rest (open);
    # ctrl=0 -> joint~0 (open), ctrl=255 -> joint~0.71 (closing). Opposite polarity
    # from Franka's actuator8 even though both share the same 0-255 ctrl range.
    open_ctrl=0.0,
    close_ctrl=255.0,
    left_finger_body="r2f85_left_pad",
    right_finger_body="r2f85_right_pad",
    # ee_site is placed at the upstream "pinch" site -- where the pads already
    # converge on a centered object -- so the residual offset is ~0, unlike
    # Franka's ee_site which sits 1.5cm above its actual fingertip contact.
    ftp_offset=0.0,
    mass_scale=0.35,
    driver_joints=("r2f85_right_driver_joint", "r2f85_left_driver_joint"),
)

END_EFFECTORS: dict[str, EndEffectorSpec] = {
    FRANKA_HAND.name: FRANKA_HAND,
    ROBOTIQ_2F85.name: ROBOTIQ_2F85,
}

DEFAULT_END_EFFECTOR = FRANKA_HAND.name


def get_spec(name: str) -> EndEffectorSpec:
    try:
        return END_EFFECTORS[name]
    except KeyError:
        valid = ", ".join(sorted(END_EFFECTORS))
        raise ValueError(f"Unknown end-effector {name!r}. Valid options: {valid}") from None


@dataclass(frozen=True)
class MountOverride:
    """Non-default wrist-mount geometry for a specific (arm, end_effector) pair.

    Most arm+gripper combos mount the gripper relative to the wrist flange
    using EndEffectorSpec.mount_pos/mount_quat/ee_site_pos unchanged (today's
    Franka-relative behavior). Kinova+Robotiq needs different geometry per
    the upstream kinova_gen3/README.md: skip Robotiq's base_mount coupling
    body entirely (it duplicates Kinova's own end-effector interface) and
    mount Robotiq's base body, plus the ee_site, at different offsets.
    """
    mount_pos: tuple[float, float, float]
    mount_quat: tuple[float, float, float, float]
    skip_base_mount: bool
    ee_site_pos: tuple[float, float, float]
    ee_site_quat: tuple[float, float, float, float]


MOUNT_OVERRIDES: dict[tuple[str, str], MountOverride] = {
    ("kinova_gen3", "robotiq_2f85"): MountOverride(
        mount_pos=(0.0, 0.0, -0.06149039),
        mount_quat=(0.0, -1.0, 1.0, 0.0),
        skip_base_mount=True,
        ee_site_pos=(0.0, 0.0, -0.181525),
        ee_site_quat=(0.0, 1.0, 0.0, 0.0),
    ),
}


def get_mount_override(arm_name: str, ee_name: str) -> MountOverride | None:
    return MOUNT_OVERRIDES.get((arm_name, ee_name))
