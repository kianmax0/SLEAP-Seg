"""Fused tracker: ByteTrack + OSNet Re-ID for stable multi-animal ID assignment.

Re-ID is optional: if torchreid is not installed, the tracker gracefully falls
back to pure ByteTrack spatial matching (still robust for most cases).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..perception.yolo_seg import Detection
from .bytetrack import ByteTracker, Track, TrackState


class FusedTracker:
    """Combines ByteTrack spatial matching with OSNet Re-ID for post-occlusion recovery.

    Workflow per frame:
    1. ByteTrack produces updated tracks (spatial association).
    2. For newly confirmed tracks (just recovered from LOST state), query ReID bank
       against previously lost track IDs and reassign if cosine similarity is high.
    3. Update ReID bank embeddings for all active tracks.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        tracking_cfg = cfg.get("tracking", {})
        reid_cfg = cfg.get("reid", {})
        device = cfg.get("device", "cpu")

        self.byte_tracker = ByteTracker(
            high_thresh=tracking_cfg.get("bytetrack_high_thresh", 0.6),
            low_thresh=tracking_cfg.get("bytetrack_low_thresh", 0.1),
            match_thresh=tracking_cfg.get("bytetrack_match_thresh", 0.8),
            max_lost_frames=tracking_cfg.get("max_lost_frames", 30),
            box_iou_weight=tracking_cfg.get("box_iou_weight", 0.6),
            mask_iou_weight=tracking_cfg.get("mask_iou_weight", 0.4),
        )

        self.reid = None
        try:
            from .reid import ReIDExtractor
            self.reid = ReIDExtractor(
                model_name=reid_cfg.get("model", "osnet_x0_25"),
                weights_path=reid_cfg.get("weights"),
                embedding_dim=reid_cfg.get("embedding_dim", 512),
                ema_alpha=reid_cfg.get("ema_alpha", 0.9),
                cosine_threshold=reid_cfg.get("cosine_threshold", 0.7),
                device=device,
            )
            print("Re-ID (OSNet) loaded successfully.")
        except (ImportError, Exception) as e:
            warnings.warn(
                f"Re-ID disabled ({e}). Tracker will use ByteTrack spatial matching only. "
                "Install torchreid to enable appearance-based re-association.",
                stacklevel=2,
            )

        # Track IDs that were LOST before this frame (for re-association)
        self._prev_lost_ids: List[int] = []

    def update(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> List[Track]:
        """Process one frame; return active tracks with stable IDs."""
        # Capture lost IDs before ByteTrack update
        prev_lost_ids = [t.track_id for t in self.byte_tracker.lost_tracks]

        active_tracks = self.byte_tracker.update(detections)

        if self.reid is not None:
            # Identify newly appeared tracks (could be re-appearances from lost)
            new_tracks = [
                t for t in active_tracks
                if t.frame_id == self.byte_tracker._frame_id and t.lost_frames == 0
            ]

            if new_tracks and prev_lost_ids:
                masks = [t.mask for t in new_tracks]
                bboxes = [t.bbox for t in new_tracks]
                embeddings = self.reid.extract(frame, masks, bboxes)

                for track, emb in zip(new_tracks, embeddings):
                    best_id, sim = self.reid.reassociate(prev_lost_ids, emb)
                    if best_id is not None:
                        old_id = track.track_id
                        track.track_id = best_id
                        prev_lost_ids.remove(best_id)
                        self.reid.bank.remove(old_id)

            # Update Re-ID bank for all active tracks
            if active_tracks:
                masks = [t.mask for t in active_tracks]
                bboxes = [t.bbox for t in active_tracks]
                embeddings = self.reid.extract(frame, masks, bboxes)
                self.reid.update_bank(
                    [t.track_id for t in active_tracks], embeddings
                )

        self._prev_lost_ids = [t.track_id for t in self.byte_tracker.lost_tracks]

        return active_tracks
