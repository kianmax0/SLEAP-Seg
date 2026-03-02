"""Quantitative validation script for SLEAP-Seg.

Compares SLEAP-Seg output against a ground-truth (GT) .slp file on two metrics:

1. ID Switches (IDS): number of times a tracked animal changes identity label
   compared to its previous frame assignment. Lower is better.

2. Occlusion-Frame Keypoint RMSE: average Euclidean error on frames where
   two animals overlap (mask IoU > threshold). Lower is better.

Usage examples:

  # Full validation against GT labels
  python scripts/validate.py \\
      --gt      /path/to/ground_truth.slp \\
      --pred    /path/to/sleap_seg_output.slp \\
      --video   /path/to/video.mp4 \\
      --report  validation_report.csv

  # Compute only ID switch metric (no video needed)
  python scripts/validate.py \\
      --gt   /path/to/gt.slp \\
      --pred /path/to/pred.slp \\
      --metric ids

  # Compare SLEAP-Seg vs pure-SLEAP on same video
  python scripts/validate.py \\
      --gt        /path/to/gt.slp \\
      --pred      /path/to/sleap_seg.slp \\
      --baseline  /path/to/pure_sleap.slp \\
      --video     /path/to/video.mp4 \\
      --report    comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────── SLP loading helpers ──────────────────────────────

def load_slp(path: str):
    """Load a SLEAP labels file, trying sleap then sleap_io."""
    try:
        import sleap  # type: ignore
        return sleap.load_file(path)
    except Exception:
        pass
    try:
        import sleap_io as sio  # type: ignore
        return sio.load_slp(path)
    except Exception as e:
        raise RuntimeError(f"Cannot load {path}: {e}")


def get_instances_per_frame(labels) -> Dict[int, List]:
    """Return {frame_idx: [instance, ...]} mapping."""
    result: Dict[int, List] = defaultdict(list)
    for lf in labels:
        fi = lf.frame_idx if hasattr(lf, "frame_idx") else int(lf.index)
        for inst in lf.instances:
            result[fi].append(inst)
    return result


def instance_to_points(inst) -> np.ndarray:
    """Convert instance to (n_nodes, 2) float32 array, NaN where invisible."""
    if hasattr(inst, "numpy"):
        arr = inst.numpy()
        return arr[:, :2].astype(np.float32)
    if hasattr(inst, "points"):
        pts = []
        for pt in inst.points:
            if hasattr(pt, "x"):
                pts.append([float(pt.x) if pt.x is not None else np.nan,
                             float(pt.y) if pt.y is not None else np.nan])
            else:
                pts.append([float(pt[0]), float(pt[1])])
        return np.array(pts, dtype=np.float32)
    return np.full((1, 2), np.nan, dtype=np.float32)


def instance_centroid(inst) -> np.ndarray:
    """Return (2,) centroid of visible keypoints."""
    pts = instance_to_points(inst)
    valid = pts[~np.any(np.isnan(pts), axis=1)]
    if valid.shape[0] == 0:
        return np.array([np.nan, np.nan])
    return valid.mean(axis=0)


# ─────────────────────────── ID Switch metric ─────────────────────────────────

def compute_id_switches(
    gt_frames: Dict[int, List],
    pred_frames: Dict[int, List],
    max_dist: float = 50.0,
) -> Dict:
    """Count ID switches using Hungarian centroid matching frame-by-frame.

    An ID switch occurs when the greedy assignment between consecutive frames
    differs from the assignment in the previous frame for the same GT instance.

    Returns a dict with:
      - id_switches: total number of switches
      - frames_with_switch: list of frame indices where switches occurred
      - total_matched_frames: number of frames where matching was attempted
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        warnings.warn("scipy not installed; using greedy assignment for IDS metric")
        linear_sum_assignment = None

    frame_ids = sorted(set(gt_frames.keys()) & set(pred_frames.keys()))
    if not frame_ids:
        return {"id_switches": 0, "frames_with_switch": [], "total_matched_frames": 0}

    # prev_assignment: gt_instance_idx -> pred_instance_idx (by centroid order)
    prev_assignment: Dict[int, int] = {}
    id_switches = 0
    frames_with_switch = []

    for fi in frame_ids:
        gt_insts = gt_frames[fi]
        pred_insts = pred_frames[fi]
        if not gt_insts or not pred_insts:
            continue

        gt_cents = np.stack([instance_centroid(inst) for inst in gt_insts])
        pred_cents = np.stack([instance_centroid(inst) for inst in pred_insts])

        # Cost matrix: Euclidean distance
        n, m = len(gt_insts), len(pred_insts)
        cost = np.full((n, m), 1e9)
        for i in range(n):
            for j in range(m):
                if not (np.any(np.isnan(gt_cents[i])) or np.any(np.isnan(pred_cents[j]))):
                    cost[i, j] = np.linalg.norm(gt_cents[i] - pred_cents[j])

        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = {int(r): int(c) for r, c in zip(row_ind, col_ind) if cost[r, c] < max_dist}
        else:
            assignment = {}
            used = set()
            for i in range(n):
                j = int(np.argmin(cost[i]))
                if cost[i, j] < max_dist and j not in used:
                    assignment[i] = j
                    used.add(j)

        # Check for switches compared to previous frame
        switched_this_frame = False
        for gt_idx, pred_idx in assignment.items():
            if gt_idx in prev_assignment and prev_assignment[gt_idx] != pred_idx:
                id_switches += 1
                switched_this_frame = True

        if switched_this_frame:
            frames_with_switch.append(fi)

        prev_assignment = assignment

    return {
        "id_switches": id_switches,
        "frames_with_switch": frames_with_switch,
        "total_matched_frames": len(frame_ids),
    }


# ─────────────────────────── Occlusion RMSE metric ────────────────────────────

def _bbox_from_points(pts: np.ndarray, pad: float = 20.0) -> Optional[np.ndarray]:
    """Bounding box (x1, y1, x2, y2) from keypoints with padding."""
    valid = pts[~np.any(np.isnan(pts), axis=1)]
    if valid.shape[0] == 0:
        return None
    x1, y1 = valid.min(axis=0) - pad
    x2, y2 = valid.max(axis=0) + pad
    return np.array([x1, y1, x2, y2])


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def find_occlusion_frames(
    gt_frames: Dict[int, List],
    iou_threshold: float = 0.15,
) -> List[int]:
    """Return frame indices where two animals overlap (bbox IoU > threshold)."""
    occluded = []
    for fi, insts in gt_frames.items():
        if len(insts) < 2:
            continue
        boxes = []
        for inst in insts:
            pts = instance_to_points(inst)
            box = _bbox_from_points(pts)
            if box is not None:
                boxes.append(box)
        if len(boxes) < 2:
            continue
        iou = _bbox_iou(boxes[0], boxes[1])
        if iou >= iou_threshold:
            occluded.append(fi)
    return occluded


def compute_keypoint_rmse(
    gt_frames: Dict[int, List],
    pred_frames: Dict[int, List],
    frame_ids: Optional[List[int]] = None,
    max_dist: float = 100.0,
) -> Dict:
    """Compute per-keypoint RMSE between GT and predicted labels.

    Args:
        gt_frames: {frame_idx: [instances]}
        pred_frames: {frame_idx: [instances]}
        frame_ids: subset of frames to evaluate (None = all shared frames)
        max_dist: max centroid distance to consider a valid GT-pred match

    Returns dict with:
      - rmse_per_node: list of RMSE per node (index order)
      - mean_rmse: overall mean RMSE
      - n_pairs: number of instance pairs evaluated
    """
    if frame_ids is None:
        frame_ids = sorted(set(gt_frames.keys()) & set(pred_frames.keys()))

    errors: List[np.ndarray] = []  # each (n_nodes,)

    for fi in frame_ids:
        gt_insts = gt_frames.get(fi, [])
        pred_insts = pred_frames.get(fi, [])
        if not gt_insts or not pred_insts:
            continue

        gt_cents = np.stack([instance_centroid(inst) for inst in gt_insts])
        pred_cents = np.stack([instance_centroid(inst) for inst in pred_insts])

        for gi, gt_inst in enumerate(gt_insts):
            if np.any(np.isnan(gt_cents[gi])):
                continue
            dists = [
                np.linalg.norm(gt_cents[gi] - pred_cents[pi])
                if not np.any(np.isnan(pred_cents[pi]))
                else np.inf
                for pi in range(len(pred_insts))
            ]
            best_pi = int(np.argmin(dists))
            if dists[best_pi] > max_dist:
                continue

            gt_pts = instance_to_points(gt_insts[gi])
            pred_pts = instance_to_points(pred_insts[best_pi])
            n = min(gt_pts.shape[0], pred_pts.shape[0])
            per_node = np.full(max(gt_pts.shape[0], pred_pts.shape[0]), np.nan)
            for k in range(n):
                gp = gt_pts[k]
                pp = pred_pts[k]
                if not (np.any(np.isnan(gp)) or np.any(np.isnan(pp))):
                    per_node[k] = np.linalg.norm(gp - pp)
            errors.append(per_node)

    if not errors:
        return {"rmse_per_node": [], "mean_rmse": np.nan, "n_pairs": 0}

    stacked = np.stack(errors)  # (n_pairs, n_nodes)
    rmse_per_node = np.nanmean(stacked, axis=0).tolist()
    mean_rmse = float(np.nanmean(stacked))
    return {
        "rmse_per_node": rmse_per_node,
        "mean_rmse": mean_rmse,
        "n_pairs": len(errors),
    }


# ─────────────────────────── Report generation ────────────────────────────────

def print_results(
    label: str,
    ids_result: Dict,
    rmse_all: Dict,
    rmse_occ: Dict,
    node_names: Optional[List[str]] = None,
) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  ID Switches:              {ids_result['id_switches']}")
    print(f"  Frames with switch:       {len(ids_result['frames_with_switch'])}")
    print(f"  Total matched frames:     {ids_result['total_matched_frames']}")
    print()
    print(f"  Overall keypoint RMSE:    {rmse_all['mean_rmse']:.2f} px  (n={rmse_all['n_pairs']})")
    print(f"  Occlusion keypoint RMSE:  {rmse_occ['mean_rmse']:.2f} px  (n={rmse_occ['n_pairs']})")

    if rmse_all['rmse_per_node'] and node_names:
        print("\n  Per-node RMSE (all frames):")
        for name, val in zip(node_names, rmse_all['rmse_per_node']):
            print(f"    {name:<15} {val:.2f} px")


def save_report(
    output_path: Path,
    rows: List[Dict],
    node_names: Optional[List[str]] = None,
) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nReport saved to: {output_path}")


# ─────────────────────────── CLI ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitative validation for SLEAP-Seg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth .slp file")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted .slp file (SLEAP-Seg output)")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Optional baseline .slp file (pure SLEAP) to compare against")
    parser.add_argument("--video", type=Path, default=None,
                        help="Original video (used for mask-based occlusion detection)")
    parser.add_argument(
        "--metric", choices=["all", "ids", "rmse"], default="all",
        help="Which metric(s) to compute. Default: all"
    )
    parser.add_argument("--iou-threshold", type=float, default=0.15,
                        help="Bbox IoU threshold to classify a frame as occluded. Default: 0.15")
    parser.add_argument("--max-dist", type=float, default=100.0,
                        help="Max centroid distance for GT-pred matching. Default: 100 px")
    parser.add_argument("--report", type=Path, default=None,
                        help="Save results to CSV file")
    args = parser.parse_args()

    print(f"Loading GT:   {args.gt}")
    gt_labels = load_slp(str(args.gt))
    gt_frames = get_instances_per_frame(gt_labels)
    print(f"  {len(gt_frames)} labeled frames, up to {max(len(v) for v in gt_frames.values())} instances")

    # Get node names
    node_names = None
    try:
        skel = gt_labels.skeleton if hasattr(gt_labels, "skeleton") else gt_labels.skeletons[0]
        if hasattr(skel, "node_names"):
            node_names = list(skel.node_names)
        elif hasattr(skel, "nodes"):
            node_names = [n.name for n in skel.nodes]
    except Exception:
        pass

    print(f"\nLoading Pred: {args.pred}")
    pred_labels = load_slp(str(args.pred))
    pred_frames = get_instances_per_frame(pred_labels)
    print(f"  {len(pred_frames)} labeled frames")

    baseline_frames = None
    if args.baseline:
        print(f"\nLoading Baseline: {args.baseline}")
        baseline_labels = load_slp(str(args.baseline))
        baseline_frames = get_instances_per_frame(baseline_labels)
        print(f"  {len(baseline_frames)} labeled frames")

    # Find occlusion frames
    occ_frames = find_occlusion_frames(gt_frames, iou_threshold=args.iou_threshold)
    print(f"\nOcclusion frames (IoU > {args.iou_threshold}): {len(occ_frames)} frames")

    report_rows = []

    def _eval(label: str, pred_f: Dict) -> Dict:
        row = {"label": label}

        if args.metric in ("all", "ids"):
            ids = compute_id_switches(gt_frames, pred_f, max_dist=args.max_dist)
            row.update({
                "id_switches": ids["id_switches"],
                "frames_with_switch": len(ids["frames_with_switch"]),
                "total_matched_frames": ids["total_matched_frames"],
            })
        else:
            ids = {"id_switches": 0, "frames_with_switch": [], "total_matched_frames": 0}
            row.update({"id_switches": "—", "frames_with_switch": "—", "total_matched_frames": "—"})

        if args.metric in ("all", "rmse"):
            rmse_all = compute_keypoint_rmse(gt_frames, pred_f, max_dist=args.max_dist)
            rmse_occ = compute_keypoint_rmse(gt_frames, pred_f, frame_ids=occ_frames, max_dist=args.max_dist)
            row.update({
                "rmse_all_frames_px": round(rmse_all["mean_rmse"], 3),
                "rmse_occlusion_frames_px": round(rmse_occ["mean_rmse"], 3),
                "n_pairs_all": rmse_all["n_pairs"],
                "n_pairs_occlusion": rmse_occ["n_pairs"],
            })
            if node_names:
                for name, val in zip(node_names, rmse_all.get("rmse_per_node", [])):
                    row[f"rmse_{name}"] = round(val, 3)
        else:
            rmse_all = {"mean_rmse": np.nan, "n_pairs": 0, "rmse_per_node": []}
            rmse_occ = {"mean_rmse": np.nan, "n_pairs": 0}

        print_results(label, ids, rmse_all, rmse_occ, node_names)
        return row

    report_rows.append(_eval("SLEAP-Seg (pred)", pred_frames))

    if baseline_frames:
        report_rows.append(_eval("Baseline (pure SLEAP)", baseline_frames))

        # Print improvement summary
        if (args.metric in ("all", "ids")
                and isinstance(report_rows[0].get("id_switches"), int)
                and isinstance(report_rows[1].get("id_switches"), int)):
            seg_ids = report_rows[0]["id_switches"]
            base_ids = report_rows[1]["id_switches"]
            if base_ids > 0:
                improvement = (base_ids - seg_ids) / base_ids * 100
                print(f"\n{'─'*40}")
                print(f"  ID Switch improvement: {improvement:+.1f}%")
                print(f"  ({base_ids} → {seg_ids} switches)")

        if (args.metric in ("all", "rmse")
                and isinstance(report_rows[0].get("rmse_occlusion_frames_px"), float)
                and isinstance(report_rows[1].get("rmse_occlusion_frames_px"), float)):
            seg_rmse = report_rows[0]["rmse_occlusion_frames_px"]
            base_rmse = report_rows[1]["rmse_occlusion_frames_px"]
            if base_rmse > 0:
                improvement = (base_rmse - seg_rmse) / base_rmse * 100
                print(f"  Occlusion RMSE improvement: {improvement:+.1f}%")
                print(f"  ({base_rmse:.2f}px → {seg_rmse:.2f}px)")

    if args.report:
        save_report(args.report, report_rows, node_names)


if __name__ == "__main__":
    main()
