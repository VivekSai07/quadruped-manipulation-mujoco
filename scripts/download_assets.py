"""Download Go2 and Franka Panda model assets from GitHub."""
from __future__ import annotations

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"

# ── Go2 assets ─────────────────────────────────────────────────────────────
GO2_RAW = "https://raw.githubusercontent.com/unitreerobotics/unitree_mujoco/main/unitree_robots/go2"

GO2_FILES = [
    "go2.xml",
    "assets/base_0.obj",
    "assets/base_1.obj",
    "assets/base_2.obj",
    "assets/base_3.obj",
    "assets/base_4.obj",
    "assets/hip_0.obj",
    "assets/hip_1.obj",
    "assets/thigh_0.obj",
    "assets/thigh_1.obj",
    "assets/thigh_mirror_0.obj",
    "assets/thigh_mirror_1.obj",
    "assets/calf_0.obj",
    "assets/calf_1.obj",
    "assets/calf_mirror_0.obj",
    "assets/calf_mirror_1.obj",
    "assets/foot.obj",
]

# ── Panda assets ────────────────────────────────────────────────────────────
PANDA_RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/franka_emika_panda"

PANDA_XML_FILES = ["panda.xml", "hand.xml"]

PANDA_MESH_FILES = [
    # Collision meshes (STL)
    "assets/link0.stl",
    "assets/link1.stl",
    "assets/link2.stl",
    "assets/link3.stl",
    "assets/link4.stl",
    "assets/link5_collision_0.obj",
    "assets/link5_collision_1.obj",
    "assets/link5_collision_2.obj",
    "assets/link6.stl",
    "assets/link7.stl",
    "assets/hand.stl",
    # Visual meshes (OBJ)
    "assets/link0_0.obj",
    "assets/link0_1.obj",
    "assets/link0_2.obj",
    "assets/link0_3.obj",
    "assets/link0_4.obj",
    "assets/link0_5.obj",
    "assets/link0_7.obj",
    "assets/link0_8.obj",
    "assets/link0_9.obj",
    "assets/link0_10.obj",
    "assets/link0_11.obj",
    "assets/link1.obj",
    "assets/link2.obj",
    "assets/link3_0.obj",
    "assets/link3_1.obj",
    "assets/link3_2.obj",
    "assets/link3_3.obj",
    "assets/link4_0.obj",
    "assets/link4_1.obj",
    "assets/link4_2.obj",
    "assets/link4_3.obj",
    "assets/link5_0.obj",
    "assets/link5_1.obj",
    "assets/link5_2.obj",
    "assets/link6_0.obj",
    "assets/link6_1.obj",
    "assets/link6_2.obj",
    "assets/link6_3.obj",
    "assets/link6_4.obj",
    "assets/link6_5.obj",
    "assets/link6_6.obj",
    "assets/link6_7.obj",
    "assets/link6_8.obj",
    "assets/link6_9.obj",
    "assets/link6_10.obj",
    "assets/link6_11.obj",
    "assets/link6_12.obj",
    "assets/link6_13.obj",
    "assets/link6_14.obj",
    "assets/link6_15.obj",
    "assets/link6_16.obj",
    "assets/link7_0.obj",
    "assets/link7_1.obj",
    "assets/link7_2.obj",
    "assets/link7_3.obj",
    "assets/link7_4.obj",
    "assets/link7_5.obj",
    "assets/link7_6.obj",
    "assets/link7_7.obj",
    "assets/hand_0.obj",
    "assets/hand_1.obj",
    "assets/hand_2.obj",
    "assets/hand_3.obj",
    "assets/hand_4.obj",
    "assets/finger_0.obj",
    "assets/finger_1.obj",
]


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name}")
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [ok]   {dest.name}")
        return True
    except urllib.error.HTTPError as e:
        print(f"  [miss] {dest.name} ({e.code})")
        return False
    except Exception as e:
        print(f"  [err]  {dest.name}: {e}")
        return False


def download_go2() -> None:
    print("\n=== Downloading Go2 assets ===")
    dest_dir = ASSETS_DIR / "go2"
    ok = failed = 0
    for rel in GO2_FILES:
        url = f"{GO2_RAW}/{rel}"
        dest = dest_dir / rel
        if download_file(url, dest):
            ok += 1
        else:
            failed += 1
    print(f"Go2: {ok} downloaded, {failed} missing (may be named differently)")


def download_panda() -> None:
    print("\n=== Downloading Panda assets ===")
    dest_dir = ASSETS_DIR / "panda"
    ok = failed = 0
    for rel in PANDA_XML_FILES + PANDA_MESH_FILES:
        url = f"{PANDA_RAW}/{rel}"
        dest = dest_dir / rel
        if download_file(url, dest):
            ok += 1
        else:
            failed += 1
    print(f"Panda: {ok} downloaded, {failed} missing")


def verify_critical() -> bool:
    critical = [
        ASSETS_DIR / "go2" / "go2.xml",
        ASSETS_DIR / "panda" / "panda.xml",
    ]
    all_ok = True
    print("\n=== Critical file check ===")
    for f in critical:
        status = "OK" if f.exists() else "MISSING"
        print(f"  [{status}] {f}")
        if not f.exists():
            all_ok = False
    return all_ok


if __name__ == "__main__":
    download_go2()
    download_panda()
    ok = verify_critical()
    sys.exit(0 if ok else 1)
