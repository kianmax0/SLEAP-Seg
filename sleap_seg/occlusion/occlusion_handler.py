"""Occlusion detection and keypoint smoothing.

When pairwise Mask IoU between any two active tracks exceeds the configured
threshold, those tracks enter OCCLUSION mode. During occlusion:
  - High-priority keypoints (nose, tail_base) are preserved if confident.
  - Low-priority keypoints are linearly interpolated between the last known
    reliable position and the first post-occlusion position.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..pose.sleap_infer import Keypoint, PoseResult
from ..tracking.bytetrack import Track


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


class _TrackOcclusionState:
    """Stores per-track occlusion bookkeeping."""

    def __init__(self, n_keypoints: int, priority_names: List[str]) -> None:
        self.n_keypoints = n_keypoints
        self.priority_names = set(priority_names)
        self.is_occluded: bool = False
        # Last reliable keypoint positions: (N_kp, 2) or None
        self.last_known: Optional[np.ndarray] = None
        self.occlusion_start_frame: int = -1
        # Buffered frames while occluded: list of (N_kp, 2) arrays
        self.occluded_buffer: List[np.ndarray] = []

    def record_reliable(self, keypoints: List[Keypoint], frame_id: int) -> None:
        coords = np.array([[kp.x, kp.y] for kp in keypoints])
        self.last_known = coords

    def enter_occlusion(self, frame_id: int) -> None:
        self.is_occluded = True
        self.occlusion_start_frame = frame_id
        self.occluded_buffer = []

    def exit_occlusion(self, post_occlusion: List[Keypoint]) -> List[np.ndarray]:
        """Return interpolated coords for each buffered frame, then reset state."""
        self.is_occluded = False
        n_frames = len(self.occluded_buffer)
        post_coords = np.array([[kp.x, kp.y] for kp in post_occlusion])

        if self.last_known is None or n_frames == 0:
            result = self.occluded_buffer
            self.occluded_buffer = []
            return result

        interpolated: List[np.ndarray] = []
        for t in range(n_frames):
            alpha = (t + 1) / (n_frames + 1)
            interp = (1.0 - alpha) * self.last_known + alpha * post_coords
            interpolated.append(interp)

        self.occluded_buffer = []
        return interpolated


class OcclusionHandler:
    """Detects pairwise occlusions and smooths affected keypoint trajectories."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        occlusion_cfg = cfg.get("occlusion", {})
        self.iou_threshold: float = occlusion_cfg.get("iou_threshold", 0.6)
        self.priority_keypoints: List[str] = occlusion_cfg.get(
            "priority_keypoints", ["nose", "tail_base"]
        )
        self.smooth_window: int = occlusion_cfg.get("smooth_window", 5)

        # { track_id: _TrackOcclusionState }
        self._states: Dict[int, _TrackOcclusionState] = {}
        # Pairs currently in occlusion
        self._occluded_pairs: Set[Tuple[int, int]] = set()

    def _get_state(self, track_id: int, n_keypoints: int) -> _TrackOcclusionState:
        if track_id not in self._states:
            self._states[track_id] = _TrackOcclusionState(
                n_keypoints, self.priority_keypoints
            )
        return self._states[track_id]

    def detect_occlusions(self, tracks: List[Track]) -> Set[int]:
        """Return set of track IDs currently involved in occlusion."""
        occluded_ids: Set[int] = set()
        new_pairs: Set[Tuple[int, int]] = set()

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                iou = _mask_iou(tracks[i].mask, tracks[j].mask)
                if iou >= self.iou_threshold:
                    pair = (tracks[i].track_id, tracks[j].track_id)
                    new_pairs.add(pair)
                    occluded_ids.add(tracks[i].track_id)
                    occluded_ids.add(tracks[j].track_id)

        self._occluded_pairs = new_pairs
        return occluded_ids

    def process(
        self,
        pose_results: List[PoseResult],
        tracks: List[Track],
        frame_id: int,
    ) -> List[PoseResult]:
        """Apply occlusion smoothing to pose results for the current frame."""
        track_map = {t.track_id: t for t in tracks}
        occluded_ids = self.detect_occlusions(tracks)

        processed: List[PoseResult] = []

        for result in pose_results:
            tid = result.track_id
            n_kp = len(result.keypoints)
            state = self._get_state(tid, n_kp)

            if tid in occluded_ids:
                if not state.is_occluded:
                    state.enter_occlusion(frame_id)
                # Buffer occluded frame coords; keep priority keypoints if confident
                coords = np.array([[kp.x, kp.y] for kp in result.keypoints])
                state.occluded_buffer.append(coords)

                # For priority keypoints: keep if trusted, else use last known
                for idx, kp in enumerate(result.keypoints):
                    if kp.name not in state.priority_keypoints or not kp.trusted:
                        if state.last_known is not None:
                            kp.x = float(state.last_known[idx, 0])
                            kp.y = float(state.last_known[idx, 1])
                        kp.trusted = False

            else:
                if state.is_occluded:
                    # Just exited occlusion — retroactively interpolate buffered frames
                    # (those frames have already been emitted; store for post-processing)
                    state.exit_occlusion(result.keypoints)

                state.record_reliable(result.keypoints, frame_id)

            processed.append(result)

        # Clean up states for tracks that disappeared
        active_ids = {t.track_id for t in tracks}
        stale = [tid for tid in self._states if tid not in active_ids]
        for tid in stale:
            del self._states[tid]

        return processed
