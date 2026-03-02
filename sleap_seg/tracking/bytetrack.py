"""ByteTrack multi-object tracker extended with Mask-IoU matching.

Implements the two-stage association from the ByteTrack paper:
  Stage 1: match high-confidence detections against active tracks
  Stage 2: match low-confidence detections against unmatched active tracks

Cost matrix: 0.6 * (1 - box_iou) + 0.4 * (1 - mask_iou)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..perception.yolo_seg import Detection


class TrackState(Enum):
    ACTIVE = auto()
    LOST = auto()
    REMOVED = auto()


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    mask: np.ndarray          # Binary mask (H, W)
    confidence: float
    state: TrackState = TrackState.ACTIVE
    lost_frames: int = 0
    frame_id: int = 0
    # Kalman state [cx, cy, w, h, vx, vy, vw, vh]
    kalman_mean: Optional[np.ndarray] = None
    kalman_cov: Optional[np.ndarray] = None


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks of identical shape."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def _cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    box_w: float = 0.6,
    mask_w: float = 0.4,
) -> np.ndarray:
    """Build (T x D) cost matrix combining box-IoU and mask-IoU."""
    cost = np.ones((len(tracks), len(detections)), dtype=np.float32)
    for ti, track in enumerate(tracks):
        for di, det in enumerate(detections):
            biou = _bbox_iou(track.bbox, det.bbox)
            miou = _mask_iou(track.mask, det.mask)
            cost[ti, di] = 1.0 - (box_w * biou + mask_w * miou)
    return cost


def _linear_assignment(cost: np.ndarray, threshold: float) -> Tuple[
    List[Tuple[int, int]], List[int], List[int]
]:
    """Solve assignment problem; return (matches, unmatched_tracks, unmatched_dets)."""
    try:
        import lap
        _, row_ind, col_ind = lap.lapjv(cost, extend_cost=True, cost_limit=threshold)
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        row_ind_arr, col_ind_arr = linear_sum_assignment(cost)
        row_ind = np.full(cost.shape[0], -1, dtype=int)
        col_ind = np.full(cost.shape[1], -1, dtype=int)
        for r, c in zip(row_ind_arr, col_ind_arr):
            row_ind[r] = c
            col_ind[c] = r

    matches: List[Tuple[int, int]] = []
    unmatched_tracks: List[int] = []
    unmatched_dets: List[int] = []

    for ti, di in enumerate(row_ind):
        if di < 0 or cost[ti, di] > threshold:
            unmatched_tracks.append(ti)
        else:
            matches.append((ti, di))

    matched_dets = {di for _, di in matches}
    for di in range(cost.shape[1]):
        if di not in matched_dets:
            unmatched_dets.append(di)

    return matches, unmatched_tracks, unmatched_dets


class ByteTracker:
    """ByteTrack with Mask-IoU extended cost matrix."""

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_lost_frames: int = 30,
        box_iou_weight: float = 0.6,
        mask_iou_weight: float = 0.4,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames
        self.box_w = box_iou_weight
        self.mask_w = mask_iou_weight

        self._next_id: int = 1
        self._active_tracks: List[Track] = []
        self._lost_tracks: List[Track] = []
        self._frame_id: int = 0

    def _new_track(self, det: Detection) -> Track:
        t = Track(
            track_id=self._next_id,
            bbox=det.bbox.copy(),
            mask=det.mask.copy(),
            confidence=det.confidence,
            frame_id=self._frame_id,
        )
        self._next_id += 1
        return t

    def _update_track(self, track: Track, det: Detection) -> None:
        track.bbox = det.bbox.copy()
        track.mask = det.mask.copy()
        track.confidence = det.confidence
        track.state = TrackState.ACTIVE
        track.lost_frames = 0
        track.frame_id = self._frame_id

    def update(self, detections: List[Detection]) -> List[Track]:
        """Process one frame of detections; return list of active Tracks with IDs."""
        self._frame_id += 1

        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        # Stage 1: match high-confidence detections against active tracks
        if self._active_tracks and high_dets:
            cost = _cost_matrix(self._active_tracks, high_dets, self.box_w, self.mask_w)
            matches, unmatched_t, unmatched_d = _linear_assignment(cost, self.match_thresh)
        else:
            matches = []
            unmatched_t = list(range(len(self._active_tracks)))
            unmatched_d = list(range(len(high_dets)))

        for ti, di in matches:
            self._update_track(self._active_tracks[ti], high_dets[di])

        # Stage 2: match low-confidence detections against remaining active tracks
        remaining_active = [self._active_tracks[i] for i in unmatched_t]
        if remaining_active and low_dets:
            cost2 = _cost_matrix(remaining_active, low_dets, self.box_w, self.mask_w)
            matches2, still_unmatched_t, _ = _linear_assignment(cost2, self.match_thresh)
            for ti, di in matches2:
                self._update_track(remaining_active[ti], low_dets[di])
                unmatched_t.remove(self._active_tracks.index(remaining_active[ti]))
        else:
            still_unmatched_t = list(range(len(remaining_active)))

        # Mark unmatched active tracks as LOST
        for i in unmatched_t:
            track = self._active_tracks[i]
            if track.state == TrackState.ACTIVE:
                track.state = TrackState.LOST
                track.lost_frames += 1
                self._lost_tracks.append(track)

        # Stage 3: try to match new high-conf detections against lost tracks
        new_det_idxs = unmatched_d
        new_dets = [high_dets[i] for i in new_det_idxs]
        if self._lost_tracks and new_dets:
            cost3 = _cost_matrix(self._lost_tracks, new_dets, self.box_w, self.mask_w)
            matches3, _, still_new = _linear_assignment(cost3, self.match_thresh)
            for ti, di in matches3:
                self._update_track(self._lost_tracks[ti], new_dets[di])
                self._active_tracks.append(self._lost_tracks[ti])
            recovered = {ti for ti, _ in matches3}
            self._lost_tracks = [
                t for i, t in enumerate(self._lost_tracks) if i not in recovered
            ]
            new_dets = [new_dets[i] for i in still_new]

        # Create new tracks for unmatched new detections
        for det in new_dets:
            self._active_tracks.append(self._new_track(det))

        # Age lost tracks; remove expired ones
        expired = []
        for t in self._lost_tracks:
            t.lost_frames += 1
            if t.lost_frames > self.max_lost_frames:
                t.state = TrackState.REMOVED
                expired.append(t)
        for t in expired:
            self._lost_tracks.remove(t)

        # Keep only ACTIVE tracks in the primary list
        self._active_tracks = [
            t for t in self._active_tracks if t.state == TrackState.ACTIVE
        ]

        return list(self._active_tracks)

    @property
    def active_tracks(self) -> List[Track]:
        return list(self._active_tracks)

    @property
    def lost_tracks(self) -> List[Track]:
        return list(self._lost_tracks)
