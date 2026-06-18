# Vendored: Kinova Gen3

- **Source**: https://github.com/google-deepmind/mujoco_menagerie
- **Path**: `kinova_gen3/`
- **Ref**: `main` branch, fetched 2026-06-17
- **License**: BSD-3-Clause (Copyright (c) 2018, Kinova inc.) — see `LICENSE` in this directory
- **Contents**: 8 STL meshes (`assets/`) only. The MJCF body/joint/actuator
  definitions were re-authored in `scripts/build_model.py` (`_kinova_arm_body_xml`)
  to integrate with this project's naming conventions (`kinova_` prefix to avoid
  colliding with Go2's own `base_link`), mass scaling, and contact
  exclude/keyframe layout — they are not a verbatim copy of `gen3.xml`.
- **Robotiq 2F-85 mount transform**: per upstream `kinova_gen3/README.md`, mounting
  Robotiq 2F-85 on Kinova skips Robotiq's `base_mount` coupling body (Kinova's
  `bracelet_link` already provides the equivalent interface) and mounts Robotiq's
  `base` body directly at `pos="0 0 -0.06149039" quat="0 -1 1 0"`, with the ee/pinch
  site shifted to `pos="0 0 -0.181525" quat="0 1 0 0"`. Encoded in
  `controllers/end_effectors.py::MOUNT_OVERRIDES[("kinova_gen3", "robotiq_2f85")]`.
