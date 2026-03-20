"""SLEAP inference wrapper supporting multiple backends.

Loading priority:
  1. sleap-nn (Python >= 3.11, PyTorch ckpt format — SLEAP >= 1.4 models)
  2. Cached NPZ (pre-computed by sleap_nn_worker.py in sleapNN conda env)
  3. sleap legacy (Python 3.10, TensorFlow h5 format — SLEAP <= 1.3 models)
  4. Graceful fallback — returns empty keypoints with a warning

The recommended workflow for Python 3.10 + .ckpt models:
  1. Pre-compute keypoints using the sleapNN env:
       conda run -n sleapNN python scripts/sleap_nn_worker.py \\
           --model /path/to/model --video /path/to/video.mp4
  2. Pass --sleap-cache /path/to/video_sleap_kps.npz to the CLI

Bottom-up models produce all instances per frame; keypoints are then
associated to ByteTrack instances by nearest-centroid matching.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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

_COST_INVALID = 1e12


def _bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
    return float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)


def _inst_centroid(inst: Any) -> np.ndarray:
    pts = []
    for pt in inst.points:
        x = float(pt.x) if hasattr(pt, "x") else float(pt[0])
        y = float(pt.y) if hasattr(pt, "y") else float(pt[1])
        if not (np.isnan(x) or np.isnan(y)):
            pts.append((x, y))
    if pts:
        return np.mean(np.asarray(pts, dtype=np.float64), axis=0)
    return np.array([np.nan, np.nan], dtype=np.float64)


def _spatial_cost_matrix(
    sleap_instances: List[Any],
    bboxes: List[np.ndarray],
) -> np.ndarray:
    n_t, n_i = len(bboxes), len(sleap_instances)
    C = np.full((n_t, n_i), _COST_INVALID, dtype=np.float64)
    track_centers = np.array([_bbox_center(bb) for bb in bboxes])
    for ti in range(n_t):
        for ii in range(n_i):
            ic = _inst_centroid(sleap_instances[ii])
            if not np.any(np.isnan(ic)):
                C[ti, ii] = float(np.linalg.norm(track_centers[ti] - ic))
    return C


def _temporal_cost_matrix(
    sleap_instances: List[Any],
    track_ids: List[int],
    prev_poses: Dict[int, PoseResult],
) -> np.ndarray:
    n_t, n_i = len(track_ids), len(sleap_instances)
    C = np.full((n_t, n_i), _COST_INVALID, dtype=np.float64)
    for ti in range(n_t):
        tid = track_ids[ti]
        prev = prev_poses.get(tid)
        if prev is None:
            continue
        for ii in range(n_i):
            inst = sleap_instances[ii]
            dists: List[float] = []
            for i, pt in enumerate(inst.points):
                if i >= len(prev.keypoints):
                    break
                x = float(pt.x) if hasattr(pt, "x") else float(pt[0])
                y = float(pt.y) if hasattr(pt, "y") else float(pt[1])
                pk = prev.keypoints[i]
                if np.isnan(x) or np.isnan(y) or np.isnan(pk.x) or np.isnan(pk.y):
                    continue
                dists.append(float(np.hypot(x - pk.x, y - pk.y)))
            if dists:
                C[ti, ii] = float(np.mean(dists))
    return C


def _hungarian_assignment(C: np.ndarray) -> Dict[int, int]:
    """Min-cost one-to-one matching; returns {track_row_idx: instance_col_idx}."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return {}

    row_ind, col_ind = linear_sum_assignment(C)
    out: Dict[int, int] = {}
    for ri, ci in zip(row_ind, col_ind):
        if C[ri, ci] < _COST_INVALID * 0.5:
            out[int(ri)] = int(ci)
    return out


def _greedy_assignment(C: np.ndarray) -> Dict[int, int]:
    """Greedy min-cost fallback when scipy is unavailable."""
    n_t, n_i = C.shape
    assigned: Dict[int, int] = {}
    CC = C.copy()
    for _ in range(min(n_t, n_i)):
        if np.all(CC >= _COST_INVALID * 0.5):
            break
        ti, ii = np.unravel_index(np.argmin(CC), CC.shape)
        if CC[ti, ii] >= _COST_INVALID * 0.5:
            break
        assigned[int(ti)] = int(ii)
        CC[ti, :] = _COST_INVALID
        CC[:, ii] = _COST_INVALID
    return assigned


def _assign_instances_to_tracks(
    sleap_instances: List[Any],
    node_names: List[str],
    bboxes: List[np.ndarray],
    track_ids: List[int],
    frame_id: int,
    peak_threshold: float = 0.2,
    assignment_mode: str = "spatial",
    prev_poses: Optional[Dict[int, PoseResult]] = None,
) -> List[PoseResult]:
    """Match SLEAP instances to tracks via Hungarian (spatial or temporal cost)."""
    n_tracks = len(track_ids)
    n_inst = len(sleap_instances)

    if n_inst == 0:
        return [_empty_pose_result(tid, node_names, frame_id) for tid in track_ids]

    C_spatial = _spatial_cost_matrix(sleap_instances, bboxes)

    if assignment_mode == "temporal" and prev_poses:
        C_temp = _temporal_cost_matrix(sleap_instances, track_ids, prev_poses)
        C = np.where(C_temp < _COST_INVALID * 0.5, C_temp, C_spatial)
    else:
        C = C_spatial

    assign = _hungarian_assignment(C)
    if not assign:
        assign = _greedy_assignment(C)

    results: List[PoseResult] = []
    for ti, tid in enumerate(track_ids):
        if ti in assign:
            inst = sleap_instances[assign[ti]]
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


def _kps_array_to_pseudo_instances(kps_frame: np.ndarray, node_names: List[str]):
    """Convert (n_inst, n_nodes, 3) array to pseudo-instance list for matcher."""

    class _Pt:
        __slots__ = ("x", "y", "score")
        def __init__(self, x, y, score):
            self.x = x; self.y = y; self.score = score

    class _Inst:
        def __init__(self, pts):
            self.points = pts

    instances = []
    for inst_kps in kps_frame:  # (n_nodes, 3)
        pts = [_Pt(float(x), float(y), float(s)) for x, y, s in inst_kps]
        instances.append(_Inst(pts))
    return instances


# ─────────────────────────── Inferencer class ─────────────────────────────────

class SLEAPInferencer:
    """Multi-backend SLEAP inferencer.

    Backends tried in order:
      1. sleap-nn direct (Python >= 3.11)
      2. NPZ cache (pre-computed by scripts/sleap_nn_worker.py)
      3. Legacy TF SLEAP (.h5 models)
      4. No-op (NaN keypoints)
    """

    def __init__(
        self,
        model_path: str,
        peak_threshold: float = 0.2,
        batch_size: int = 4,
        device: str = "cpu",
        cache_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.peak_threshold = peak_threshold
        self.batch_size = batch_size
        self.device = device
        self._node_names: List[str] = []
        self._predictor: Any = None
        self._backend: str = "none"
        self._cache: Optional[np.ndarray] = None  # (n_frames, max_inst, n_nodes, 3)

        if cache_path:
            self._load_npz_cache(cache_path)
        else:
            self._load_model()

    def _load_model(self) -> None:
        model_dir = Path(self.model_path)
        has_ckpt = (model_dir / "best.ckpt").exists()
        has_h5 = (model_dir / "best_model.h5").exists() or any(model_dir.glob("*.h5"))

        # 1. Try sleap-nn directly (Python >= 3.11)
        if has_ckpt and sys.version_info >= (3, 11):
            try:
                self._load_sleap_nn(model_dir)
                return
            except Exception as e:
                warnings.warn(f"sleap-nn direct load failed: {e}", stacklevel=2)

        # 2. Try legacy sleap TF (.h5 models)
        if has_h5 or not has_ckpt:
            try:
                self._load_legacy_sleap(model_dir)
                return
            except Exception as e:
                warnings.warn(f"Legacy SLEAP load failed: {e}", stacklevel=2)

        # 3. Graceful fallback
        self._node_names = self._read_nodes_from_labels(model_dir)
        if has_ckpt:
            warnings.warn(
                f"SLEAP model at '{self.model_path}' could not be loaded.\n"
                "  → Model requires sleap-nn (Python 3.11+).\n"
                "  → Pre-compute keypoints with:\n"
                "      conda run -n sleapNN python scripts/sleap_nn_worker.py \\\n"
                f"          --model {self.model_path} --video YOUR_VIDEO.mp4\n"
                "  → Then pass --sleap-cache <path>.npz to the CLI.",
                stacklevel=2,
            )
        self._backend = "none"

    def _load_sleap_nn(self, model_dir: Path) -> None:
        """Load via sleap-nn BottomUpPredictor (Python >= 3.11 only)."""
        from sleap_nn.inference.predictors import BottomUpPredictor  # type: ignore
        from omegaconf import OmegaConf  # type: ignore

        # Work around sleap-nn 0.1.0 bug: preprocess_config must not be None
        preprocess_config = None
        training_yaml = model_dir / "training_config.yaml"
        if training_yaml.exists():
            try:
                cfg = OmegaConf.load(str(training_yaml))
                preprocess_config = cfg.data_config.preprocessing
            except Exception:
                pass

        self._predictor = BottomUpPredictor.from_trained_models(
            bottomup_ckpt_path=str(model_dir),
            device=self.device,
            batch_size=self.batch_size,
            preprocess_config=preprocess_config,
        )
        self._node_names = _get_node_names_from_predictor(self._predictor, model_dir)
        self._backend = "sleap-nn"
        print(f"SLEAP-NN loaded ({len(self._node_names)} nodes): {self._node_names}")

    def _load_legacy_sleap(self, model_dir: Path) -> None:
        """Load via legacy SLEAP TF API."""
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        import sleap  # type: ignore
        predictor = sleap.load_model(str(model_dir))
        self._predictor = predictor
        self._node_names = [n.name for n in predictor.model.skeletons[0].nodes]
        self._backend = "sleap-legacy"
        print(f"SLEAP-Legacy loaded ({len(self._node_names)} nodes): {self._node_names}")

    def _load_npz_cache(self, cache_path: str) -> None:
        """Load pre-computed keypoints from NPZ file."""
        path = Path(cache_path)
        if not path.exists():
            warnings.warn(f"SLEAP cache not found: {cache_path}", stacklevel=2)
            self._backend = "none"
            return
        data = np.load(path, allow_pickle=True)
        self._cache = data["keypoints"]       # (n_frames, max_inst, n_nodes, 3)
        self._node_names = list(data["node_names"])
        self._backend = "cache"
        print(
            f"SLEAP cache loaded: {self._cache.shape[0]} frames, "
            f"{len(self._node_names)} nodes: {self._node_names}"
        )

    def _read_nodes_from_labels(self, model_dir: Path) -> List[str]:
        """Read skeleton node names from the training .slp labels file."""
        try:
            os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
            import sleap  # type: ignore
            slp_files = list(model_dir.glob("labels_train_gt_*.slp"))
            if slp_files:
                labels = sleap.load_file(str(slp_files[0]))
                return [n.name for n in labels.skeleton.nodes]
        except Exception:
            pass
        return []

    @property
    def node_names(self) -> List[str]:
        return self._node_names

    @property
    def backend(self) -> str:
        return self._backend

    def infer(
        self,
        frame: np.ndarray,
        bboxes: List[np.ndarray],
        track_ids: List[int],
        frame_id: int = 0,
        *,
        assignment_mode: str = "spatial",
        prev_poses: Optional[Dict[int, PoseResult]] = None,
    ) -> List[PoseResult]:
        """Run inference on a full frame and associate results to track IDs.

        assignment_mode:
          - ``spatial``: bbox-centroid vs skeleton-centroid (Hungarian).
          - ``temporal``: mean keypoint displacement vs previous frame (Hungarian),
            with spatial fallback where temporal cost is undefined.
        """
        if not track_ids:
            return []

        if self._backend == "none" or (self._predictor is None and self._cache is None):
            return [_empty_pose_result(tid, self._node_names, frame_id) for tid in track_ids]

        try:
            if self._backend == "cache":
                return self._infer_from_cache(
                    bboxes, track_ids, frame_id, assignment_mode, prev_poses
                )
            elif self._backend == "sleap-nn":
                return self._infer_sleap_nn(
                    frame, bboxes, track_ids, frame_id, assignment_mode, prev_poses
                )
            elif self._backend == "sleap-legacy":
                return self._infer_legacy(
                    frame, bboxes, track_ids, frame_id, assignment_mode, prev_poses
                )
        except Exception as e:
            warnings.warn(f"Inference failed on frame {frame_id}: {e}", stacklevel=2)

        return [_empty_pose_result(tid, self._node_names, frame_id) for tid in track_ids]

    def _infer_from_cache(
        self,
        bboxes: List[np.ndarray],
        track_ids: List[int],
        frame_id: int,
        assignment_mode: str = "spatial",
        prev_poses: Optional[Dict[int, PoseResult]] = None,
    ) -> List[PoseResult]:
        """Return keypoints from pre-computed NPZ cache."""
        assert self._cache is not None
        n_frames = self._cache.shape[0]
        if frame_id >= n_frames:
            return [_empty_pose_result(tid, self._node_names, frame_id) for tid in track_ids]

        kps_frame = self._cache[frame_id]  # (max_inst, n_nodes, 3)
        # Filter out all-NaN instances
        valid = [
            kps_frame[i]
            for i in range(kps_frame.shape[0])
            if not np.all(np.isnan(kps_frame[i, :, :2]))
        ]
        if not valid:
            return [_empty_pose_result(tid, self._node_names, frame_id) for tid in track_ids]

        instances = _kps_array_to_pseudo_instances(np.stack(valid), self._node_names)
        return _assign_instances_to_tracks(
            instances,
            self._node_names,
            bboxes,
            track_ids,
            frame_id,
            self.peak_threshold,
            assignment_mode=assignment_mode,
            prev_poses=prev_poses,
        )

    def _infer_sleap_nn(
        self,
        frame: np.ndarray,
        bboxes: List[np.ndarray],
        track_ids: List[int],
        frame_id: int,
        assignment_mode: str = "spatial",
        prev_poses: Optional[Dict[int, PoseResult]] = None,
    ) -> List[PoseResult]:
        """Per-frame inference using sleap-nn BottomUpPredictor (Python 3.11+)."""
        import sleap_io as sio  # type: ignore
        import tempfile

        # Save frame to temp video file for predictor
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tmp_path = tf.name

        try:
            import cv2
            out = cv2.VideoWriter(
                tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0])
            )
            out.write(frame)
            out.release()

            video = sio.Video(filename=tmp_path)
            self._predictor.make_pipeline(inference_object=video)
            labels = self._predictor.predict(make_labels=True)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if not labels:
            return [_empty_pose_result(tid, self._node_names, frame_id) for tid in track_ids]

        lf = labels[0]
        instances = list(lf.instances) if hasattr(lf, "instances") else []
        if not instances:
            return [_empty_pose_result(tid, self._node_names, frame_id) for tid in track_ids]

        class _Pt:
            __slots__ = ("x", "y", "score")
            def __init__(self, x, y, score):
                self.x = x; self.y = y; self.score = score

        class _Inst:
            def __init__(self, pts):
                self.points = pts

        pseudo = []
        n_nodes = len(self._node_names)
        for inst in instances:
            arr = inst.numpy() if hasattr(inst, "numpy") else None
            if arr is None:
                continue
            pts = []
            for i in range(min(n_nodes, arr.shape[0])):
                x, y = float(arr[i, 0]), float(arr[i, 1])
                s = float(arr[i, 2]) if arr.shape[1] > 2 else 1.0
                pts.append(_Pt(x, y, s))
            pseudo.append(_Inst(pts))

        return _assign_instances_to_tracks(
            pseudo,
            self._node_names,
            bboxes,
            track_ids,
            frame_id,
            self.peak_threshold,
            assignment_mode=assignment_mode,
            prev_poses=prev_poses,
        )

    def _infer_legacy(
        self,
        frame: np.ndarray,
        bboxes: List[np.ndarray],
        track_ids: List[int],
        frame_id: int,
        assignment_mode: str = "spatial",
        prev_poses: Optional[Dict[int, PoseResult]] = None,
    ) -> List[PoseResult]:
        """Per-frame inference using legacy TF SLEAP."""
        import sleap  # type: ignore
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video = sleap.Video.from_numpy(gray[np.newaxis, :, :, np.newaxis])
        labeled_frames = self._predictor.predict(video)

        all_instances = []
        if labeled_frames and labeled_frames[0].instances:
            all_instances = list(labeled_frames[0].instances)

        return _assign_instances_to_tracks(
            all_instances,
            self._node_names,
            bboxes,
            track_ids,
            frame_id,
            self.peak_threshold,
            assignment_mode=assignment_mode,
            prev_poses=prev_poses,
        )


# ─────────────────── Batch pre-compute helper ─────────────────────────────────

def precompute_sleap_cache(
    video_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    device: str = "cpu",
    conda_env: str = "sleapNN",
) -> Optional[str]:
    """Run the sleap_nn_worker in the sleapNN conda env to pre-compute keypoints.

    Returns the output .npz path on success, None on failure.
    This function is called from the CLI when --sleap-cache is not provided
    but the sleapNN environment is available.
    """
    python_exe = _find_conda_python(conda_env)
    if python_exe is None:
        warnings.warn(
            f"sleapNN conda env not found. Cannot pre-compute SLEAP cache.\n"
            f"Install it with: conda create -n sleapNN python=3.11 -y && "
            f"conda run -n sleapNN pip install git+https://github.com/talmolab/sleap-nn.git",
            stacklevel=2,
        )
        return None

    worker = _find_worker_script()
    vp = Path(video_path)
    out = Path(output_path) if output_path else vp.parent / (vp.stem + "_sleap_kps.npz")

    if out.exists():
        print(f"SLEAP cache already exists: {out}")
        return str(out)

    print(f"Pre-computing SLEAP keypoints (this may take a few minutes)...")
    print(f"  Video: {vp.name}")
    print(f"  Model: {model_path}")
    print(f"  Output: {out}")

    result = subprocess.run(
        [
            python_exe, worker,
            "--model", model_path,
            "--video", str(vp),
            "--output", str(out),
            "--device", device,
        ],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        warnings.warn(f"sleap_nn_worker failed (exit {result.returncode})", stacklevel=2)
        return None

    if out.exists():
        print(f"SLEAP cache saved: {out}")
        return str(out)

    return None


# ─────────────────── Helper functions ─────────────────────────────────────────

def _get_node_names_from_predictor(predictor, model_dir: Path) -> List[str]:
    """Try various attributes to get skeleton node names."""
    for attr in ("inference_model", "torch_model", "model"):
        sub = getattr(predictor, attr, None)
        if sub is not None:
            names = _extract_names(sub)
            if names:
                return names

    names = _extract_names(predictor)
    if names:
        return names

    # Fallback: read from training labels file
    try:
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        import sleap  # type: ignore
        slp_files = list(model_dir.glob("labels_train_gt_*.slp"))
        if slp_files:
            labels = sleap.load_file(str(slp_files[0]))
            return [n.name for n in labels.skeleton.nodes]
    except Exception:
        pass
    return []


def _extract_names(obj) -> List[str]:
    for attr in ("skeleton", "skeletons"):
        skel = getattr(obj, attr, None)
        if skel is None:
            continue
        if isinstance(skel, (list, tuple)):
            skel = skel[0]
        if hasattr(skel, "node_names"):
            return list(skel.node_names)
        if hasattr(skel, "nodes"):
            return [n.name for n in skel.nodes]
    return []


def _find_worker_script() -> str:
    candidates = [
        Path(__file__).parent.parent.parent / "scripts" / "sleap_nn_worker.py",
        Path.cwd() / "scripts" / "sleap_nn_worker.py",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("sleap_nn_worker.py not found at scripts/sleap_nn_worker.py")


def _find_conda_python(env_name: str) -> Optional[str]:
    """Find the Python executable for a named conda environment."""
    conda_exe = shutil.which("conda")
    if conda_exe:
        try:
            result = subprocess.run(
                [conda_exe, "run", "-n", env_name, "which", "python"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                py = result.stdout.strip()
                if py and Path(py).exists():
                    return py
        except Exception:
            pass

    home = Path.home()
    for prefix in [
        home / "miniconda3" / "envs" / env_name / "bin" / "python",
        home / "anaconda3" / "envs" / env_name / "bin" / "python",
        home / "opt" / "anaconda3" / "envs" / env_name / "bin" / "python",
        Path("/opt/homebrew/Caskroom/miniconda/base/envs") / env_name / "bin" / "python",
        Path("/opt/miniconda3/envs") / env_name / "bin" / "python",
        Path("/usr/local/Caskroom/miniconda/base/envs") / env_name / "bin" / "python",
    ]:
        if prefix.exists():
            return str(prefix)
    return None
