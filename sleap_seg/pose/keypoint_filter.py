"""Mask-constraint filter and Kalman-based keypoint interpolation.

Two responsibilities:
1. MaskConstraint  – marks keypoints outside the instance mask as untrusted.
2. KalmanKeypoint  – one Kalman filter per (track_id, keypoint_idx); predicts
                     positions for untrusted/missing keypoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

from .sleap_infer import Keypoint, PoseResult


# ──────────────────────────── Mask Constraint ─────────────────────────────────

def apply_mask_constraint(
    pose_result: PoseResult,
    mask: np.ndarray,
) -> PoseResult:
    """Set keypoints outside the instance mask to trusted=False."""
    if mask is None or mask.sum() == 0:
        return pose_result

    # Pre-compute contours once for the mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for kp in pose_result.keypoints:
        if not kp.trusted or np.isnan(kp.x) or np.isnan(kp.y):
            continue

        inside = False
        for contour in contours:
            dist = cv2.pointPolygonTest(contour, (float(kp.x), float(kp.y)), False)
            if dist >= 0:
                inside = True
                break

        if not inside:
            kp.trusted = False

    return pose_result


# ──────────────────────────── Kalman Filter ───────────────────────────────────

def _make_kalman(process_noise: float, measurement_noise: float) -> KalmanFilter:
    """Create a constant-velocity 2-D Kalman filter. State: [x, y, vx, vy]."""
    kf = KalmanFilter(dim_x=4, dim_z=2)

    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    kf.R *= measurement_noise
    kf.Q *= process_noise

    return kf


class KalmanKeypoint:
    """Per-(track_id, keypoint_idx) Kalman filter for position interpolation."""

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # { (track_id, kp_idx): KalmanFilter }
        self._filters: Dict[Tuple[int, int], KalmanFilter] = {}
        self._initialised: Dict[Tuple[int, int], bool] = {}

    def _get_or_create(self, track_id: int, kp_idx: int) -> KalmanFilter:
        key = (track_id, kp_idx)
        if key not in self._filters:
            self._filters[key] = _make_kalman(self.process_noise, self.measurement_noise)
            self._initialised[key] = False
        return self._filters[key]

    def predict_and_update(
        self,
        pose_result: PoseResult,
    ) -> PoseResult:
        """For each keypoint: predict next state; update if trusted; fill if untrusted."""
        tid = pose_result.track_id

        for idx, kp in enumerate(pose_result.keypoints):
            kf = self._get_or_create(tid, idx)
            key = (tid, idx)

            if self._initialised[key]:
                kf.predict()

            if kp.trusted and not (np.isnan(kp.x) or np.isnan(kp.y)):
                measurement = np.array([[kp.x], [kp.y]], dtype=float)
                if not self._initialised[key]:
                    kf.x = np.array([[kp.x], [kp.y], [0.0], [0.0]], dtype=float)
                    self._initialised[key] = True
                kf.update(measurement)
            else:
                # Keypoint is untrusted — use Kalman prediction to fill
                if self._initialised[key]:
                    kp.x = float(kf.x[0])
                    kp.y = float(kf.x[1])
                # If not yet initialised, leave as NaN

        return pose_result

    def remove_track(self, track_id: int, n_keypoints: int) -> None:
        """Clean up filters when a track is removed."""
        for idx in range(n_keypoints):
            key = (track_id, idx)
            self._filters.pop(key, None)
            self._initialised.pop(key, None)


# ──────────────────────────── Combined Filter ─────────────────────────────────

class KeypointFilter:
    """Applies mask constraint then Kalman interpolation to each PoseResult."""

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        self.kalman = KalmanKeypoint(process_noise, measurement_noise)

    def filter(
        self,
        pose_result: PoseResult,
        mask: Optional[np.ndarray],
    ) -> PoseResult:
        if mask is not None:
            pose_result = apply_mask_constraint(pose_result, mask)
        pose_result = self.kalman.predict_and_update(pose_result)
        return pose_result

    def remove_track(self, track_id: int, n_keypoints: int) -> None:
        self.kalman.remove_track(track_id, n_keypoints)
