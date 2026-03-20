"""Occlusion detection and keypoint smoothing.

When pairwise Mask IoU between any two active tracks exceeds the configured
threshold, those tracks enter OCCLUSION mode. During occlusion:
  - High-priority keypoints (nose, tail_base) are preserved if confident.
  - Low-priority keypoints are linearly interpolated between the last known
    reliable position and the first post-occlusion position.

V2: supports FrameState (merged blob / pairwise), risk streak, and long-occlusion NaN.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..pose.sleap_infer import Keypoint, PoseResult
from ..state.frame_state import FrameState
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
        self.priority_keypoints = set(priority_names)
        self.is_occluded: bool = False
        self.last_known: Optional[np.ndarray] = None
        self.occlusion_start_frame: int = -1
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


def _nullify_pose_result(result: PoseResult) -> PoseResult:
    for kp in result.keypoints:
        kp.x = float("nan")
        kp.y = float("nan")
        kp.trusted = False
    return result


class OcclusionHandler:
    """Detects occlusions and smooths affected keypoint trajectories."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        occlusion_cfg = cfg.get("occlusion", {})
        pipeline_cfg = cfg.get("pipeline", {})
        self.iou_threshold: float = occlusion_cfg.get("iou_threshold", 0.6)
        self.priority_keypoints: List[str] = occlusion_cfg.get(
            "priority_keypoints", ["nose", "tail_base"]
        )
        self.smooth_window: int = occlusion_cfg.get("smooth_window", 5)
        self.nan_after_frames: int = int(
            pipeline_cfg.get("occlusion", {}).get("nan_after_frames", 0)
        )

        self._states: Dict[int, _TrackOcclusionState] = {}
        self._occluded_pairs: Set[Tuple[int, int]] = set()

    def _get_state(self, track_id: int, n_keypoints: int) -> _TrackOcclusionState:
        if track_id not in self._states:
            self._states[track_id] = _TrackOcclusionState(
                n_keypoints, self.priority_keypoints
            )
        return self._states[track_id]

    def detect_occlusions(self, tracks: List[Track]) -> Set[int]:
        """Return set of track IDs currently involved in pairwise mask occlusion."""
        occluded_ids: Set[int] = set()
        new_pairs: Set[Tuple[int, int]] = set()

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                iou = _mask_iou(tracks[i].mask, tracks[j].mask)
                if iou >= self.iou_threshold:
                    new_pairs.add((tracks[i].track_id, tracks[j].track_id))
                    occluded_ids.add(tracks[i].track_id)
                    occluded_ids.add(tracks[j].track_id)

        self._occluded_pairs = new_pairs
        return occluded_ids

    def _resolve_occluded_ids(
        self,
        frame_state: Optional[FrameState],
        pose_results: List[PoseResult],
        tracks: List[Track],
    ) -> Set[int]:
        if frame_state == FrameState.MERGED_BLOB:
            return {r.track_id for r in pose_results}
        if frame_state == FrameState.PAIRWISE_OCCLUSION:
            return self.detect_occlusions(tracks)
        if frame_state == FrameState.PEACE:
            return set()
        # Backward compat: no frame_state — use pairwise IoU only
        return self.detect_occlusions(tracks)

    def process(
        self,
        pose_results: List[PoseResult],
        tracks: List[Track],
        frame_id: int,
        frame_state: Optional[FrameState] = None,
        risk_streak: int = 0,
    ) -> List[PoseResult]:
        """Apply occlusion smoothing; nullify all keypoints if risk streak exceeds limit."""
        if (
            self.nan_after_frames > 0
            and risk_streak > self.nan_after_frames
            and pose_results
        ):
            return [_nullify_pose_result(r) for r in pose_results]

        occluded_ids = self._resolve_occluded_ids(frame_state, pose_results, tracks)

        processed: List[PoseResult] = []

        for result in pose_results:
            tid = result.track_id
            n_kp = len(result.keypoints)
            state = self._get_state(tid, n_kp)

            if tid in occluded_ids:
                if not state.is_occluded:
                    state.enter_occlusion(frame_id)
                coords = np.array([[kp.x, kp.y] for kp in result.keypoints])
                state.occluded_buffer.append(coords)

                for idx, kp in enumerate(result.keypoints):
                    if kp.name not in state.priority_keypoints or not kp.trusted:
                        if state.last_known is not None:
                            kp.x = float(state.last_known[idx, 0])
                            kp.y = float(state.last_known[idx, 1])
                        kp.trusted = False

            else:
                if state.is_occluded:
                    state.exit_occlusion(result.keypoints)

                state.record_reliable(result.keypoints, frame_id)

            processed.append(result)

        active_ids = {t.track_id for t in tracks} | {r.track_id for r in pose_results}
        stale = [tid for tid in self._states if tid not in active_ids]
        for tid in stale:
            del self._states[tid]

        return processed
