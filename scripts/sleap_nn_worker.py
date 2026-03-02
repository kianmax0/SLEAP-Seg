"""Batch SLEAP-NN inference worker.

This script runs in the sleapNN (Python 3.11) conda environment.
It processes an entire video and saves all keypoints to a .npz cache file,
which is then read by the main sleapSeg pipeline.

Usage (called automatically by the main pipeline, or manually):

  conda run -n sleapNN python scripts/sleap_nn_worker.py \\
      --model /path/to/model_dir \\
      --video /path/to/video.mp4 \\
      --output /path/to/keypoints.npz

  # Smoke test (no video needed):
  conda run -n sleapNN python scripts/sleap_nn_worker.py \\
      --model /path/to/model_dir --mode test

Output .npz format:
  keypoints:  float32 (n_frames, max_instances, n_nodes, 3)   [x, y, score]
  node_names: str array of shape (n_nodes,)
  frame_count: int scalar
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def load_predictor(model_dir: Path, device: str = "cpu"):
    """Load sleap-nn BottomUpPredictor, auto-reading preprocess_config from training_config.yaml."""
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

    predictor = BottomUpPredictor.from_trained_models(
        bottomup_ckpt_path=str(model_dir),
        device=device,
        batch_size=4,
        preprocess_config=preprocess_config,
    )
    return predictor


def get_node_names(predictor) -> List[str]:
    """Extract skeleton node names from predictor."""
    # Try inference_model chain
    for attr in ("inference_model", "torch_model", "model"):
        sub = getattr(predictor, attr, None)
        if sub is None:
            continue
        names = _extract_names(sub)
        if names:
            return names

    names = _extract_names(predictor)
    if names:
        return names

    # Fallback: read from training labels
    model_dir = getattr(predictor, "_bottomup_ckpt_path", None)
    if model_dir:
        return _read_names_from_labels(Path(model_dir))
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


def _read_names_from_labels(model_dir: Path) -> List[str]:
    try:
        import sleap_io as sio  # type: ignore
        import glob
        slp_files = glob.glob(str(model_dir / "labels_train_gt_*.slp"))
        if slp_files:
            labels = sio.load_slp(slp_files[0])
            return [n.name for n in labels.skeleton.nodes]
    except Exception:
        pass
    return []


def _instances_to_array(instances, n_nodes: int) -> np.ndarray:
    """Convert instance list to float32 array (n_instances, n_nodes, 3)."""
    if not instances:
        return np.full((0, n_nodes, 3), np.nan, dtype=np.float32)

    rows = []
    for inst in instances:
        pts = np.full((n_nodes, 3), np.nan, dtype=np.float32)
        if hasattr(inst, "numpy"):
            arr = inst.numpy()  # (n_nodes, 2) from sleap_io
            n = min(arr.shape[0], n_nodes)
            pts[:n, :2] = arr[:n]
            pts[:n, 2] = 1.0
        elif hasattr(inst, "score"):
            # sio.PredictedInstance: has .points dict or .numpy()
            try:
                arr = inst.numpy()
                n = min(arr.shape[0], n_nodes)
                pts[:n, :2] = arr[:n, :2]
                if arr.shape[1] >= 3:
                    pts[:n, 2] = arr[:n, 2]
                else:
                    pts[:n, 2] = float(inst.score) if inst.score is not None else 1.0
            except Exception:
                pass
        elif hasattr(inst, "points"):
            for j, pt in enumerate(inst.points):
                if j >= n_nodes:
                    break
                if hasattr(pt, "x"):
                    pts[j, 0] = float(pt.x) if pt.x is not None else np.nan
                    pts[j, 1] = float(pt.y) if pt.y is not None else np.nan
                    pts[j, 2] = float(pt.score) if hasattr(pt, "score") and pt.score is not None else 1.0
                else:
                    pts[j, :2] = [float(pt[0]), float(pt[1])]
                    pts[j, 2] = 1.0
        rows.append(pts)

    return np.stack(rows, axis=0)  # (n_instances, n_nodes, 3)


def process_video(
    predictor,
    video_path: Path,
    output_path: Path,
    device: str = "cpu",
    max_frames: Optional[int] = None,
) -> dict:
    """Run batch inference on entire video, save to .npz.

    Returns summary dict with node_names, frame_count, etc.
    """
    import sleap_io as sio  # type: ignore

    node_names = get_node_names(predictor)
    n_nodes = len(node_names) if node_names else 1

    # Use sleap_io.Video to load the video
    video = sio.Video(filename=str(video_path))
    n_frames = video.shape[0] if video.shape else None

    print(f"Video: {video_path.name}")
    if n_frames:
        print(f"  Frames: {n_frames}, Nodes: {n_nodes}")

    # Run predictor on entire video
    predictor.make_pipeline(inference_object=video)
    t0 = time.time()
    labels = predictor.predict(make_labels=True)
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s ({len(labels)} labeled frames)")

    # Build dense array indexed by frame_idx
    actual_n_frames = max(lf.frame_idx + 1 for lf in labels) if labels else 0
    if n_frames:
        actual_n_frames = max(actual_n_frames, n_frames)

    # Find max instances across all frames
    max_instances = max((len(lf.instances) for lf in labels), default=0)
    if max_instances == 0:
        max_instances = 1

    keypoints = np.full(
        (actual_n_frames, max_instances, n_nodes, 3), np.nan, dtype=np.float32
    )

    for lf in labels:
        fi = lf.frame_idx
        if fi >= actual_n_frames:
            continue
        for ii, inst in enumerate(lf.instances[:max_instances]):
            inst_arr = _instances_to_array([inst], n_nodes)
            if inst_arr.shape[0] > 0:
                keypoints[fi, ii] = inst_arr[0]

    np.savez_compressed(
        output_path,
        keypoints=keypoints,
        node_names=np.array(node_names, dtype=object),
        frame_count=actual_n_frames,
    )
    print(f"  Saved to: {output_path}")
    return {
        "node_names": node_names,
        "frame_count": actual_n_frames,
        "max_instances": max_instances,
        "output_path": str(output_path),
    }


def test_mode(model_dir: Path, device: str = "cpu") -> None:
    """Smoke test: load model and verify node names."""
    print(f"Loading model from {model_dir}...")
    predictor = load_predictor(model_dir, device=device)
    print(f"  Predictor type: {type(predictor).__name__}")
    node_names = get_node_names(predictor)
    print(f"  Nodes ({len(node_names)}): {node_names}")
    print("sleap-nn worker ready.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch sleap-nn inference worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model directory")
    parser.add_argument("--video", type=Path, default=None, help="Path to input video file")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output .npz file path (default: <video_stem>_sleap_kps.npz)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device: 'cpu', 'mps', 'cuda'. Default: cpu"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Process only the first N frames (for testing)"
    )
    parser.add_argument(
        "--mode", choices=["infer", "test"], default="infer",
        help="'infer' = process video; 'test' = smoke test only"
    )

    args = parser.parse_args()

    if args.mode == "test":
        test_mode(args.model, device=args.device)
        return

    if args.video is None:
        print("Error: --video is required for infer mode")
        sys.exit(1)

    if not args.video.exists():
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    output_path = args.output or args.video.parent / (args.video.stem + "_sleap_kps.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model}...")
    predictor = load_predictor(args.model, device=args.device)

    summary = process_video(
        predictor=predictor,
        video_path=args.video,
        output_path=output_path,
        device=args.device,
        max_frames=args.max_frames,
    )

    print(f"\nDone. Summary: {summary}")


if __name__ == "__main__":
    main()
