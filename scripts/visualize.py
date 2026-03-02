"""Debug visualization: overlay masks, track IDs, keypoints, and occlusion events."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Colour palette (BGR) for up to 20 track IDs
_PALETTE = [
    (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))
    for _ in range(20)
]


def _track_color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    track_id: int,
    alpha: float = 0.35,
) -> np.ndarray:
    color = _track_color(track_id)
    overlay = frame.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    track_id: int,
    occluded: bool = False,
) -> np.ndarray:
    color = _track_color(track_id)
    x1, y1, x2, y2 = bbox.astype(int)
    thickness = 2
    style = cv2.LINE_AA
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, style)

    label = f"ID:{track_id}"
    if occluded:
        label += " [OCC]"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3, style)

    cv2.putText(
        frame, label, (x1, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, style
    )
    return frame


def draw_keypoints(
    frame: np.ndarray,
    keypoints: list,
    track_id: int,
    skeleton_edges: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    color = _track_color(track_id)
    pts = []

    for kp in keypoints:
        if np.isnan(kp.x) or np.isnan(kp.y):
            pts.append(None)
            continue
        cx, cy = int(kp.x), int(kp.y)
        pts.append((cx, cy))
        dot_color = color if kp.trusted else (0, 0, 200)
        cv2.circle(frame, (cx, cy), 4, dot_color, -1, cv2.LINE_AA)

    if skeleton_edges and pts:
        for (i, j) in skeleton_edges:
            if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                cv2.line(frame, pts[i], pts[j], color, 1, cv2.LINE_AA)

    return frame


def run_visualizer(
    video_path: str,
    seg_model: str,
    sleap_model: str,
    config: str = "config/default.yaml",
    output: Optional[str] = None,
    device: str = "cpu",
    max_frames: Optional[int] = None,
) -> None:
    import yaml
    from sleap_seg.perception.seg_backend import build_backend
    from sleap_seg.tracking.tracker import FusedTracker
    from sleap_seg.pose.sleap_infer import SLEAPInferencer
    from sleap_seg.pose.keypoint_filter import KeypointFilter
    from sleap_seg.occlusion.occlusion_handler import OcclusionHandler

    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg["perception"]["yolo_model"] = seg_model
    cfg["pose"]["sleap_model"] = sleap_model
    cfg["device"] = device

    seg = build_backend(cfg)
    tracker = FusedTracker(cfg)
    sleap_infer = SLEAPInferencer(
        model_path=sleap_model,
        peak_threshold=cfg["pose"].get("peak_threshold", 0.2),
        batch_size=cfg["pose"].get("batch_size", 8),
        device=device,
    )
    kp_filter = KeypointFilter(
        process_noise=cfg["kalman"].get("process_noise", 1e-2),
        measurement_noise=cfg["kalman"].get("measurement_noise", 1e-1),
    )
    occlusion_handler = OcclusionHandler(cfg)

    cap = cv2.VideoCapture(video_path)
    writer: Optional[cv2.VideoWriter] = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_id >= max_frames):
            break

        detections = seg.detect(frame)
        tracks = tracker.update(frame, detections)
        mask_by_id = {t.track_id: t.mask for t in tracks}

        bboxes = [t.bbox for t in tracks]
        tids = [t.track_id for t in tracks]
        pose_results = sleap_infer.infer(frame, bboxes, tids, frame_id)
        pose_results = occlusion_handler.process(pose_results, tracks, frame_id)

        occluded_ids = {
            tid
            for pair in occlusion_handler._occluded_pairs
            for tid in pair
        }

        vis = frame.copy()
        for track in tracks:
            vis = draw_mask(vis, track.mask, track.track_id)
            vis = draw_bbox(vis, track.bbox, track.track_id, track.track_id in occluded_ids)

        for result in pose_results:
            mask = mask_by_id.get(result.track_id)
            result = kp_filter.filter(result, mask)
            vis = draw_keypoints(vis, result.keypoints, result.track_id)

        cv2.putText(
            vis, f"Frame {frame_id}", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        if writer:
            writer.write(vis)
        else:
            cv2.imshow("SLEAP-Seg Debug", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

    cap.release()
    if writer:
        writer.release()
        print(f"Saved visualization to {output}")
    else:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="SLEAP-Seg debug visualization")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--seg-model", required=True, help="YOLOv8-seg weights path")
    parser.add_argument("--sleap-model", required=True, help="SLEAP model path")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None, help="Save visualization to file instead of showing")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames processed")
    args = parser.parse_args()

    run_visualizer(
        video_path=args.video,
        seg_model=args.seg_model,
        sleap_model=args.sleap_model,
        config=args.config,
        output=args.output,
        device=args.device,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
