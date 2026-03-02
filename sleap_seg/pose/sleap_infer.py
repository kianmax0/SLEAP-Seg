"""SLEAP inference wrapper: crop-based keypoint prediction per tracked instance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class Keypoint:
    """Single predicted keypoint."""

    __slots__ = ("x", "y", "score", "name", "trusted")

    def __init__(
        self,
        x: float,
        y: float,
        score: float,
        name: str = "",
        trusted: bool = True,
    ) -> None:
        self.x = x
        self.y = y
        self.score = score
        self.name = name
        self.trusted = trusted


class PoseResult:
    """Pose prediction for one tracked instance in full-frame coordinates."""

    def __init__(
        self,
        track_id: int,
        keypoints: List[Keypoint],
        frame_id: int,
    ) -> None:
        self.track_id = track_id
        self.keypoints = keypoints
        self.frame_id = frame_id

    @property
    def as_array(self) -> np.ndarray:
        """Return (N_kp, 3) array of [x, y, score]."""
        return np.array([[kp.x, kp.y, kp.score] for kp in self.keypoints])


class SLEAPInferencer:
    """Loads a SLEAP .pkg.slp model and runs crop-based inference.

    Each tracked animal is cropped from the full frame using its bounding box,
    inference is run on the crop, and keypoints are projected back to full-frame
    coordinates.
    """

    def __init__(
        self,
        model_path: str,
        peak_threshold: float = 0.2,
        batch_size: int = 8,
        device: str = "cpu",
    ) -> None:
        import sleap

        self.peak_threshold = peak_threshold
        self.batch_size = batch_size

        self._predictor = sleap.load_model(model_path, batch_size=batch_size)
        self._node_names: List[str] = [
            n.name for n in self._predictor.model.skeleton.nodes
        ]

    def _crop_frame(
        self, frame: np.ndarray, bbox: np.ndarray, pad: int = 10
    ) -> Tuple[np.ndarray, int, int]:
        """Crop a padded region from the frame; return crop and top-left offset."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]) - pad)
        y1 = max(0, int(bbox[1]) - pad)
        x2 = min(w, int(bbox[2]) + pad)
        y2 = min(h, int(bbox[3]) + pad)
        return frame[y1:y2, x1:x2], x1, y1

    def infer(
        self,
        frame: np.ndarray,
        bboxes: List[np.ndarray],
        track_ids: List[int],
        frame_id: int = 0,
    ) -> List[PoseResult]:
        """Run SLEAP on each crop; return PoseResult list in full-frame coords."""
        import sleap

        results: List[PoseResult] = []

        # Process in batches
        for batch_start in range(0, len(bboxes), self.batch_size):
            batch_bboxes = bboxes[batch_start : batch_start + self.batch_size]
            batch_ids = track_ids[batch_start : batch_start + self.batch_size]

            crops_and_offsets = [
                self._crop_frame(frame, bbox) for bbox in batch_bboxes
            ]
            crops = [c[0] for c in crops_and_offsets]
            offsets = [(c[1], c[2]) for c in crops_and_offsets]

            for crop, (ox, oy), tid, bbox in zip(
                crops, offsets, batch_ids, batch_bboxes
            ):
                # Build a single-frame SLEAP video/labels input
                labeled_frame = sleap.load_frames(crop[np.newaxis, ...])
                predictions = self._predictor.predict(labeled_frame)

                keypoints: List[Keypoint] = []
                if predictions and predictions[0].instances:
                    # Take the highest-confidence instance in the crop
                    best_instance = max(
                        predictions[0].instances,
                        key=lambda inst: float(
                            np.nanmean([pt.score or 0 for pt in inst.points])
                        ),
                    )
                    for node_name, pt in zip(
                        self._node_names, best_instance.points
                    ):
                        x_full = float(pt.x) + ox if not np.isnan(pt.x) else float("nan")
                        y_full = float(pt.y) + oy if not np.isnan(pt.y) else float("nan")
                        score = float(pt.score) if pt.score is not None else 0.0
                        trusted = score >= self.peak_threshold and not (
                            np.isnan(x_full) or np.isnan(y_full)
                        )
                        keypoints.append(
                            Keypoint(
                                x=x_full,
                                y=y_full,
                                score=score,
                                name=node_name,
                                trusted=trusted,
                            )
                        )
                else:
                    # No detection in crop — all keypoints untrusted
                    keypoints = [
                        Keypoint(x=float("nan"), y=float("nan"), score=0.0, name=n, trusted=False)
                        for n in self._node_names
                    ]

                results.append(PoseResult(track_id=tid, keypoints=keypoints, frame_id=frame_id))

        return results
