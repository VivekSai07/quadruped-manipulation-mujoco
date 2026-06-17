# Vendored: Robotiq 2F-85

- **Source**: https://github.com/google-deepmind/mujoco_menagerie
- **Path**: `robotiq_2f85/`
- **Ref**: `main` branch, fetched 2026-06-17
- **License**: BSD-2-Clause (Copyright (c) 2013, ROS-Industrial) — see `LICENSE` in this directory
- **Contents**: 8 STL meshes (`assets/`) only. The MJCF body/joint/tendon/actuator
  definitions were re-authored in `scripts/build_model.py` (`_build_robotiq_gripper_xml`)
  to integrate with this project's naming conventions, mass scaling, and contact
  exclude/keyframe layout — they are not a verbatim copy of `2f85.xml`.
