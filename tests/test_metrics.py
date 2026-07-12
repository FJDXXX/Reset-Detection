from __future__ import annotations

import numpy as np
import pytest

from src.loader import load_episode
from src.metrics import (
    max_joint_error,
    gripper_error,
    ee_position_error,
    ee_orientation_error,
    compute_episode_metrics,
)
from src.config import Thresholds


class TestMetrics:
    def test_max_joint_error_zero(self):
        j = np.zeros(6)
        assert max_joint_error(j, j) == 0.0

    def test_max_joint_error_positive(self):
        j1 = np.zeros(6)
        j2 = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        assert max_joint_error(j1, j2) == 0.2

    def test_gripper_error_zero(self):
        assert gripper_error(0.0, 0.0) == 0.0

    def test_gripper_error_positive(self):
        assert gripper_error(0.0, 0.05) == 0.05

    def test_ee_position_error(self):
        ee1 = np.array([0.0, 0.0, 0.0])
        ee2 = np.array([1.0, 0.0, 0.0])
        assert ee_position_error(ee1, ee2) == 1.0

    def test_ee_orientation_error(self):
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        q2 = np.array([0.0, 0.0, 0.0, 1.0])
        assert ee_orientation_error(q1, q2) == 0.0

    def test_ee_orientation_error_90deg(self):
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        q2 = np.array([0.7071068, 0.0, 0.0, 0.7071068])
        err = ee_orientation_error(q1, q2)
        assert abs(err - np.pi / 2) < 1e-4

    def test_compute_metrics_good(self, good_episode):
        ep = load_episode(good_episode)
        m = compute_episode_metrics(ep, "right_arm", Thresholds())
        assert m["max_joint_error"] == 0.0
        assert m["gripper_error"] == 0.0
        assert m["joint_fail"] is False
        assert m["gripper_fail"] is False

    def test_compute_metrics_bad_joint(self, bad_joint_episode):
        ep = load_episode(bad_joint_episode)
        thresholds = Thresholds(max_joint_error=0.15)
        m = compute_episode_metrics(ep, "right_arm", thresholds)
        assert m["max_joint_error"] > 0.15
        assert m["joint_fail"] is True

    def test_compute_metrics_bad_gripper(self, bad_gripper_episode):
        ep = load_episode(bad_gripper_episode)
        thresholds = Thresholds(gripper_error=0.01)
        m = compute_episode_metrics(ep, "right_arm", thresholds)
        assert m["gripper_error"] > 0.01
        assert m["gripper_fail"] is True
