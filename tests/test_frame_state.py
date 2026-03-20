"""Tests for V2 frame state classification."""

from __future__ import annotations

import numpy as np

from sleap_seg.state.frame_state import (
    FrameState,
    compute_frame_state,
    mask_iou_pairwise,
)


class _Det:
    def __init__(self):
        pass


class _Track:
    def __init__(self, mask: np.ndarray, tid: int = 1):
        self.mask = mask
        self.track_id = tid
        self.bbox = np.array([0, 0, 10, 10], dtype=np.float32)


def test_mask_iou_identical():
    m = np.ones((10, 10), dtype=bool)
    assert mask_iou_pairwise(m, m) == 1.0


def test_merged_blob_state():
    cfg = {"pipeline": {"expected_mice": 2}, "occlusion": {"iou_threshold": 0.6}}
    dets = [_Det()]
    tracks = [_Track(np.ones((5, 5), dtype=bool))]
    assert compute_frame_state(dets, tracks, cfg) == FrameState.MERGED_BLOB


def test_peace_two_detections():
    cfg = {"pipeline": {"expected_mice": 2}, "occlusion": {"iou_threshold": 0.6}}
    dets = [_Det(), _Det()]
    m = np.zeros((20, 20), dtype=bool)
    m[:10, :10] = True
    m2 = np.zeros((20, 20), dtype=bool)
    m2[10:, 10:] = True
    tracks = [_Track(m, 1), _Track(m2, 2)]
    assert compute_frame_state(dets, tracks, cfg) == FrameState.PEACE


def test_pairwise_occlusion():
    cfg = {"pipeline": {"expected_mice": 2}, "occlusion": {"iou_threshold": 0.3}}
    dets = [_Det(), _Det()]
    m = np.zeros((20, 20), dtype=bool)
    m[5:15, 5:15] = True
    m2 = m.copy()
    tracks = [_Track(m, 1), _Track(m2, 2)]
    assert compute_frame_state(dets, tracks, cfg) == FrameState.PAIRWISE_OCCLUSION
