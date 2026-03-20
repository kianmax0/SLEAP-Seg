"""Tests for Hungarian assignment helpers in sleap_infer."""

from __future__ import annotations

import numpy as np

from sleap_seg.pose.sleap_infer import (
    Keypoint,
    PoseResult,
    _hungarian_assignment,
    _spatial_cost_matrix,
    _temporal_cost_matrix,
)


class _Pt:
    def __init__(self, x: float, y: float, score: float = 1.0):
        self.x = x
        self.y = y
        self.score = score


class _Inst:
    def __init__(self, points):
        self.points = points


def test_hungarian_prefers_diagonal_minima():
    C = np.array([[1.0, 1e13], [1e13, 1.0]], dtype=np.float64)
    assign = _hungarian_assignment(C)
    assert assign.get(0) == 0
    assert assign.get(1) == 1


def test_spatial_cost_matrix_shape():
    inst0 = _Inst([_Pt(0.0, 0.0), _Pt(10.0, 0.0)])
    inst1 = _Inst([_Pt(100.0, 100.0), _Pt(110.0, 100.0)])
    bboxes = [
        np.array([0.0, 0.0, 20.0, 20.0], dtype=np.float32),
        np.array([90.0, 90.0, 120.0, 120.0], dtype=np.float32),
    ]
    C = _spatial_cost_matrix([inst0, inst1], bboxes)
    assert C.shape == (2, 2)
    assert C[0, 0] < C[0, 1]
    assert C[1, 1] < C[1, 0]


def test_temporal_cost_matrix_shape_and_prefers_match():
    inst0 = _Inst([_Pt(0.0, 0.0), _Pt(5.0, 0.0)])
    inst1 = _Inst([_Pt(100.0, 0.0), _Pt(105.0, 0.0)])
    prev0 = PoseResult(
        track_id=1,
        keypoints=[
            Keypoint(0.0, 0.0, 1.0, "a", True),
            Keypoint(5.0, 0.0, 1.0, "b", True),
        ],
        frame_id=0,
    )
    prev1 = PoseResult(
        track_id=2,
        keypoints=[
            Keypoint(100.0, 0.0, 1.0, "a", True),
            Keypoint(105.0, 0.0, 1.0, "b", True),
        ],
        frame_id=0,
    )
    C = _temporal_cost_matrix(
        [inst0, inst1],
        [1, 2],
        {1: prev0, 2: prev1},
    )
    assert C.shape == (2, 2)
    assert C[0, 0] < 1e6
    assert C[1, 1] < 1e6
