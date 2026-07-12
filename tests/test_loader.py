from __future__ import annotations

import pytest

from src.loader import load_episode, Episode


class TestLoader:
    def test_load_good_episode(self, good_episode):
        ep = load_episode(good_episode)
        assert isinstance(ep, Episode)
        assert "right_arm" in ep.arms
        assert ep.arms["right_arm"].arm_name == "right_arm"

    def test_start_pose(self, good_episode):
        ep = load_episode(good_episode)
        start = ep.get_start_pose("right_arm")
        assert start["joint_positions"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert start["gripper"] == 0.0

    def test_end_pose(self, bad_joint_episode):
        ep = load_episode(bad_joint_episode)
        end = ep.get_end_pose("right_arm")
        assert end["joint_positions"][0] == 0.3

    def test_num_frames(self, good_episode):
        ep = load_episode(good_episode)
        assert len(ep.arms["right_arm"].frames) == 100

    def test_missing_directory(self):
        with pytest.raises(NotADirectoryError):
            load_episode("/nonexistent/path")

    def test_missing_meta(self, tmp_path):
        d = tmp_path / "episode_no_meta"
        d.mkdir()
        (d / "actions").mkdir()
        with pytest.raises(FileNotFoundError):
            load_episode(d)
