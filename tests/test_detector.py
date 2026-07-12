from __future__ import annotations

import pytest

from src.loader import load_episode
from src.detector import check_episode
from src.config import Config, Thresholds


class TestDetector:
    def test_good_episode_passes(self, good_episode, default_config):
        ep = load_episode(good_episode)
        result = check_episode(ep, default_config)
        assert result["passed"] is True
        assert result["fail_reasons"] == []

    def test_bad_joint_fails(self, bad_joint_episode, default_config):
        ep = load_episode(bad_joint_episode)
        result = check_episode(ep, default_config)
        assert result["passed"] is False

    def test_bad_gripper_fails(self, bad_gripper_episode, default_config):
        ep = load_episode(bad_gripper_episode)
        result = check_episode(ep, default_config)
        assert result["passed"] is False

    def test_fail_reason_contains_metric(self, bad_joint_episode, default_config):
        ep = load_episode(bad_joint_episode)
        result = check_episode(ep, default_config)
        assert any("joint_error" in r for r in result["fail_reasons"])

    def test_detection_mode_all(self, bad_gripper_episode):
        ep = load_episode(bad_gripper_episode)
        config = Config(
            arms=["right_arm"],
            thresholds=Thresholds(max_joint_error=0.15, gripper_error=0.01),
            detection_mode="all",
        )
        result = check_episode(ep, config)
        assert result["passed"] is True
