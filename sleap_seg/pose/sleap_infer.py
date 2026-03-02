"""SLEAP inference wrapper supporting multiple backends.

Loading priority:
  1. sleap-nn (Python >= 3.11, PyTorch ckpt format — SLEAP >= 1.4 models)
  2. sleap legacy (Python 3.10, TensorFlow h5 format — SLEAP <= 1.3 models)
  3. Graceful fallback — returns empty keypoints with a warning

Bottom-up models (detected by best.ckpt presence) are inferred on the full frame;
keypoints are then associated to ByteTrack instances by nearest-centroid matching.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────── Data structures ──────────────────────────────────

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
        return np.array([[kp.x, kp.y, kp.score] for kp in self.keypoints])

    def centroid(self) -> Tuple[float, float]:
        """Mean (x, y) of trusted, non-NaN keypoints."""
        xs = [kp.x for kp in self.keypoints if kp.trusted and not np.isnan(kp.x)]
        ys = [kp.y for kp in self.keypoints if kp.trusted and not np.isnan(kp.y)]
        if xs and ys:
            return float(np.mean(xs)), float(np.mean(ys))
        return float("nan"), float("nan")


def _empty_pose_result(
    track_id: int, node_names: List[str], frame_id: int
) -> PoseResult:
    """PoseResult with all NaN keypoints (used when inference is unavailable)."""
    return PoseResult(
        track_id=track_id,
        keypoints=[
            Keypoint(x=float("nan"), y=float("nan"), score=0.0, name=n, trusted=False)
            for n in node_names
        ],
        frame_id=frame_id,
    )


# ─────────────────── Association: SLEAP instances → track IDs ─────────────────

def _bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
    return float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)


def _assign_instances_to_tracks(
    sleap_instances: List[Any],     # list of objects with .points and .score
    node_names: List[str],
    bboxes: List[np.ndarray],
    track_ids: List[int],
    frame_id: int,
    peak_threshold: float = 0.2,
) -> List[PoseResult]:
    """Greedy nearest-centroid matching of SLEAP instances to ByteTrack boxes."""
    n_tracks = len(track_ids)
    n_inst = len(sleap_instances)

    if n_inst == 0:
        return [_empty_pose_result(tid, node_names, frame_id) for tid in track_ids]

    track_centers = np.array([_bbox_center(bb) for bb in bboxes])  # (T, 2)

    # Build instance centroids from trusted keypoints
    inst_centers = []
    for inst in sleap_instances:
        pts = []
        for pt in inst.points:
            x = float(pt.x) if hasattr(pt, "x") else float(pt[0])
            y = float(pt.y) if hasattr(pt, "y") else float(pt[1])
            if not (np.isnan(x) or np.isnan(y)):
                pts.append((x, y))
        if pts:
            inst_centers.append(np.mean(pts, axis=0))
        else:
            inst_centers.append(np.array([float("nan"), float("nan")]))
    inst_centers = np.array(inst_centers)  # (I, 2)

    # Greedy matching by Euclidean distance
    assigned: Dict[int, int] = {}   # track_idx -> inst_idx
    used_inst = set()
    dist_matrix = np.full((n_tracks, n_inst), np.inf)
    for ti in range(n_tracks):
        for ii in range(n_inst):
            if not np.any(np.isnan(inst_centers[ii])):
                dist_matrix[ti, ii] = np.linalg.norm(track_centers[ti] - inst_centers[ii])

    for _ in range(min(n_tracks, n_inst)):
        if np.all(np.isinf(dist_matrix)):
            break
        ti, ii = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        assigned[ti] = ii
        used_inst.add(ii)
        dist_matrix[ti, :] = np.inf
        dist_matrix[:, ii] = np.inf

    results: List[PoseResult] = []
    for ti, tid in enumerate(track_ids):
        if ti in assigned:
            inst = sleap_instances[assigned[ti]]
            keypoints = []
            for node_name, pt in zip(node_names, inst.points):
                x = float(pt.x) if hasattr(pt, "x") else float(pt[0])
                y = float(pt.y) if hasattr(pt, "y") else float(pt[1])
                score = float(pt.score) if hasattr(pt, "score") and pt.score is not None else 0.0
                trusted = score >= peak_threshold and not (np.isnan(x) or np.isnan(y))
                keypoints.append(Keypoint(x=x, y=y, score=score, name=node_name, trusted=trusted))
            results.append(PoseResult(track_id=tid, keypoints=keypoints, frame_id=frame_id))
        else:
            results.append(_empty_pose_result(tid, node_names, frame_id))

    return results


# ─────────────────────────── Inferencer class ─────────────────────────────────

class SLEAPInferencer:
    """Multi-backend SLEAP inferencer.

    Tries sleap-nn (PyTorch, requires Python ≥ 3.11), then falls back to the
    legacy TensorFlow-based sleap predictor, and finally to graceful no-op mode.
    """

    def __init__(
        self,
        model_path: str,
        peak_threshold: float = 0.2,
        batch_size: int = 4,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.peak_threshold = peak_threshold
        self.batch_size = batch_size
        self.device = device
        self._node_names: List[str] = []
        self._predictor: Any = None
        self._backend: str = "none"

        self._load_model()

    def _load_model(self) -> None:
        model_dir = Path(self.model_path)
        has_ckpt = (model_dir / "best.ckpt").exists()
        has_h5 = (model_dir / "best_model.h5").exists() or any(model_dir.glob("*.h5"))

        # Try sleap-nn (PyTorch, for .ckpt models)
        if has_ckpt:
            try:
                self._load_sleap_nn(model_dir)
                return
            except Exception as e:
                warnings.warn(
                    f"sleap-nn load failed ({e}). Trying legacy SLEAP...",
                    stacklevel=2,
                )

        # Try legacy sleap TF (for .h5 models)
        if has_h5 or not has_ckpt:
            try:
                self._load_legacy_sleap(model_dir)
                return
            except Exception as e:
                warnings.warn(
                    f"Legacy SLEAP load failed ({e}). SLEAP inference disabled.",
                    stacklevel=2,
                )

        # Graceful fallback: read skeleton from .slp labels file
        self._node_names = self._read_nodes_from_labels(model_dir)
        if self._node_names:
            warnings.warn(
                f"SLEAP model could not be loaded from {self.model_path}. "
                "Keypoints will be NaN. To enable SLEAP inference with a .ckpt model, "
                "use Python >= 3.11 (required by sleap-nn).",
                stacklevel=2,
            )
        self._backend = "none"

    def _load_sleap_nn(self, model_dir: Path) -> None:
        """Load via sleap-nn (Python >= 3.11 only)."""
        import sys
        if sys.version_info < (3, 11):
            raise ImportError("sleap-nn requires Python >= 3.11")
        from sleap_nn.inference.bottomup import BottomUpInferenceModel  # type: ignore
        self._predictor = BottomUpInferenceModel.load_model(str(model_dir))
        self._node_names = list(self._predictor.skeleton.node_names)
        self._backend = "sleap-nn"
        print(f"SLEAP-NN loaded ({len(self._node_names)} nodes): {self._node_names}")

    def _load_legacy_sleap(self, model_dir: Path) -> None:
        """Load via legacy SLEAP TF API."""
        import os
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        import sleap
        predictor = sleap.load_model(str(model_dir))
        self._predictor = predictor
        self._node_names = [n.name for n in predictor.model.skeletons[0].nodes]
        self._backend = "sleap-legacy"
        print(f"SLEAP-Legacy loaded ({len(self._node_names)} nodes): {self._node_names}")

    def _read_nodes_from_labels(self, model_dir: Path) -> List[str]:
        """Read skeleton node names from the training .slp labels file."""
        try:
            import os
            os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
            import sleap
            slp_files = list(model_dir.glob("labels_train_gt_*.slp"))
            if slp_files:
                labels = sleap.load_file(str(slp_files[0]))
                return [n.name for n in labels.skeleton.nodes]
        except Exception:
            pass
        return []

    def infer(
        self,
        frame: np.ndarray,
        bboxes: List[np.ndarray],
        track_ids: List[int],
        frame_id: int = 0,
    ) -> List[PoseResult]:
        """Run inference on a full frame and associate results to track IDs."""
        if not track_ids:
            return []

        if self._backend == "none" or self._predictor is None:
            return [_empty_pose_result(tid, self._node_names, frame_id) for tid in track_ids]

        try:
            if self._backend == "sleap-nn":
                return self._infer_sleap_nn(frame, bboxes, track_ids, frame_id)
            elif self._backend == "sleap-legacy":
                return self._infer_legacy(frame, bboxes, track_ids, frame_id)
        except Exception as e:
            warnings.warn(f"Inference failed on frame {frame_id}: {e}", stacklevel=2)

        return [_empty_pose_result(tid, self._node_names, frame_id) for tid in track_ids]

    def _infer_sleap_nn(
        self,
        frame: np.ndarray,
        bboxes: List[np.ndarray],
        track_ids: List[int],
        frame_id: int,
    ) -> List[PoseResult]:
        import torch
        frame_rgb = frame[:, :, ::-1].copy()  # BGR -> RGB
        tensor = torch.from_numpy(frame_rgb).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self._predictor(tensor)

        instances = output.get("instances", [])
        return _assign_instances_to_tracks(
            instances, self._node_names, bboxes, track_ids, frame_id, self.peak_threshold
        )

    def _infer_legacy(
        self,
        frame: np.ndarray,
        bboxes: List[np.ndarray],
        track_ids: List[int],
        frame_id: int,
    ) -> List[PoseResult]:
        import sleap
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video = sleap.Video.from_numpy(gray[np.newaxis, :, :, np.newaxis])
        labeled_frames = self._predictor.predict(video)

        all_instances = []
        if labeled_frames and labeled_frames[0].instances:
            all_instances = list(labeled_frames[0].instances)

        return _assign_instances_to_tracks(
            all_instances, self._node_names, bboxes, track_ids, frame_id, self.peak_threshold
        )
