"""Tests for scripts/run_simulation.py helpers that don't need MuJoCo."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_simulation import _resolve_video_path


class TestResolveVideoPath:
    def test_requested_path_passes_through(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        requested = "custom_output.mp4"
        assert _resolve_video_path(requested, "franka", "franka") == requested

    def test_requested_path_passes_through_even_if_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        existing = tmp_path / "already_here.mp4"
        existing.write_bytes(b"")
        assert _resolve_video_path(str(existing), "franka", "franka") == str(existing)

    def test_auto_names_with_arm_and_end_effector(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _resolve_video_path(None, "kinova_gen3", "robotiq_2f85")
        assert result == str(Path("media") / "simulation_recording_kinova_gen3_robotiq_2f85.mp4")

    def test_auto_naming_creates_media_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _resolve_video_path(None, "franka", "franka")
        assert (tmp_path / "media").is_dir()

    def test_auto_naming_does_not_clobber_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        media_dir = Path("media")
        media_dir.mkdir()
        (media_dir / "simulation_recording_franka_franka.mp4").write_bytes(b"")

        result = _resolve_video_path(None, "franka", "franka")
        assert result == str(media_dir / "simulation_recording_franka_franka_2.mp4")

    def test_auto_naming_increments_past_multiple_existing_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        media_dir = Path("media")
        media_dir.mkdir()
        (media_dir / "simulation_recording_franka_franka.mp4").write_bytes(b"")
        (media_dir / "simulation_recording_franka_franka_2.mp4").write_bytes(b"")
        (media_dir / "simulation_recording_franka_franka_3.mp4").write_bytes(b"")

        result = _resolve_video_path(None, "franka", "franka")
        assert result == str(media_dir / "simulation_recording_franka_franka_4.mp4")
