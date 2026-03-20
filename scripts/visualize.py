"""Debug visualization: overlay masks, track IDs, keypoints, and occlusion events."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


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
    sleap_model: Optional[str],
    config: str = "config/default.yaml",
    output: Optional[str] = None,
    device: str = "cpu",
    max_frames: Optional[int] = None,
) -> None:
    import yaml
    from sleap_seg.perception.seg_backend import build_backend
    from sleap_seg.pipeline import Pipeline
    from sleap_seg.state.frame_state import FrameState
    from sleap_seg.tracking.tracker import FusedTracker

    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg["perception"]["yolo_model"] = seg_model
    cfg["pose"]["sleap_model"] = sleap_model or ""
    cfg["device"] = device

    if sleap_model:
        pipeline = Pipeline(cfg)
        seg = tracker = None
    else:
        pipeline = None
        seg = build_backend(cfg)
        tracker = FusedTracker(cfg)

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

        if pipeline is not None:
            pose_results, _ = pipeline.process_frame(frame, frame_id)
            tracks = pipeline.last_active_tracks
            st = pipeline.last_frame_state
        else:
            detections = seg.detect(frame)
            tracks = tracker.update(frame, detections)
            pose_results = []
            st = None

        mask_by_id = {t.track_id: t.mask for t in tracks}

        occluded_ids = set()
        if st in (FrameState.MERGED_BLOB, FrameState.PAIRWISE_OCCLUSION):
            occluded_ids = {t.track_id for t in tracks}

        vis = frame.copy()
        for track in tracks:
            vis = draw_mask(vis, track.mask, track.track_id)
            vis = draw_bbox(vis, track.bbox, track.track_id, track.track_id in occluded_ids)

        for result in pose_results:
            mask = mask_by_id.get(result.track_id)
            if mask is None and tracks:
                mask = tracks[0].mask
            vis = draw_keypoints(vis, result.keypoints, result.track_id)

        state_lbl = f" | {st.name}" if st is not None else ""
        cv2.putText(
            vis, f"Frame {frame_id}{state_lbl}", (10, 28),
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
    parser.add_argument("--sleap-model", default=None, help="SLEAP model path (optional)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None, help="Save visualization to file instead of showing")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames processed")
    parser.add_argument("--skip-sleap", action="store_true", help="Skip SLEAP inference (show seg+tracking only)")
    args = parser.parse_args()

    sleap_model = None if args.skip_sleap else args.sleap_model
    run_visualizer(
        video_path=args.video,
        seg_model=args.seg_model,
        sleap_model=sleap_model,
        config=args.config,
        output=args.output,
        device=args.device,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
