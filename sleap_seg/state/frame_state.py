"""Per-frame pipeline state (peace vs merged blob vs pairwise occlusion)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..perception.yolo_seg import Detection
    from ..tracking.bytetrack import Track


class FrameState(Enum):
    """High-level regime for assignment and post-processing."""

    PEACE = auto()
    """Two (or more) distinct detections; spatial assignment is reliable."""

    MERGED_BLOB = auto()
    """Single YOLO detection while expecting multiple animals (e.g. overlapping mice)."""

    PAIRWISE_OCCLUSION = auto()
    """Multiple tracks with strongly overlapping instance masks."""


def mask_iou_pairwise(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary masks (same shape)."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def compute_frame_state(
    detections: List[Any],
    tracks: List[Any],
    cfg: Dict[str, Any],
) -> FrameState:
    """Classify the current frame for V2 routing.

    Priority:
      1. MERGED_BLOB if exactly one detection and expected_mice >= 2
      2. PAIRWISE_OCCLUSION if >= 2 tracks and any pair exceeds mask IoU threshold
      3. PEACE otherwise
    """
    pipeline_cfg = cfg.get("pipeline", {})
    expected_mice = int(pipeline_cfg.get("expected_mice", 2))
    occ_cfg = cfg.get("occlusion", {})
    iou_threshold = float(occ_cfg.get("iou_threshold", 0.6))

    if len(detections) == 1 and expected_mice >= 2:
        return FrameState.MERGED_BLOB

    if len(tracks) >= 2:
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                iou = mask_iou_pairwise(tracks[i].mask, tracks[j].mask)
                if iou >= iou_threshold:
                    return FrameState.PAIRWISE_OCCLUSION

    return FrameState.PEACE


@dataclass
class PoseTrackView:
    """Per-track geometry for pose assignment (may include ghost second ID in merged mode)."""

    track_id: int
    bbox: np.ndarray
    mask: np.ndarray
    is_ghost: bool = False


def build_pose_track_views(
    active_tracks: List[Any],
    frame_state: FrameState,
    sticky_track_ids: List[int],
    expected_mice: int,
) -> List[PoseTrackView]:
    """Expand ByteTrack output into logical views for SLEAP assignment.

    In MERGED_BLOB with one physical track and two sticky IDs, duplicate bbox/mask
    for the ghost track so temporal assignment can split skeletons.
    """
    if not active_tracks:
        return []

    if frame_state == FrameState.MERGED_BLOB and len(active_tracks) == 1:
        t0 = active_tracks[0]
        if len(sticky_track_ids) >= 2:
            return [
                PoseTrackView(
                    track_id=sticky_track_ids[0],
                    bbox=t0.bbox,
                    mask=t0.mask,
                    is_ghost=False,
                ),
                PoseTrackView(
                    track_id=sticky_track_ids[1],
                    bbox=t0.bbox.copy(),
                    mask=t0.mask.copy(),
                    is_ghost=True,
                ),
            ]
        # Degraded: only one sticky ID — single view
        return [
            PoseTrackView(
                track_id=t0.track_id,
                bbox=t0.bbox,
                mask=t0.mask,
                is_ghost=False,
            ),
        ]

    # PEACE or PAIRWISE: one view per active track
    return [
        PoseTrackView(
            track_id=t.track_id,
            bbox=t.bbox,
            mask=t.mask,
            is_ghost=False,
        )
        for t in active_tracks
    ]
