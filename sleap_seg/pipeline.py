"""Main pipeline orchestrator: frame-loop connecting all modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .perception.seg_backend import build_backend
from .tracking.tracker import FusedTracker
from .pose.sleap_infer import SLEAPInferencer
from .pose.keypoint_filter import KeypointFilter
from .occlusion.occlusion_handler import OcclusionHandler
from .export.slp_exporter import SLPExporter


class Pipeline:
    """End-to-end SLEAP-Seg pipeline.

    Orchestrates:
      Perception → Tracking → Occlusion detection →
      SLEAP inference → Keypoint filtering → Export
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        device = cfg.get("device", "cpu")
        pose_cfg = cfg.get("pose", {})
        export_cfg = cfg.get("export", {})
        kalman_cfg = cfg.get("kalman", {})

        self.seg_backend = build_backend(cfg)
        self.tracker = FusedTracker(cfg)
        self.sleap = SLEAPInferencer(
            model_path=pose_cfg.get("sleap_model", ""),
            peak_threshold=pose_cfg.get("peak_threshold", 0.2),
            batch_size=pose_cfg.get("batch_size", 8),
            device=device,
            cache_path=pose_cfg.get("sleap_cache"),
        )
        self.kp_filter = KeypointFilter(
            process_noise=kalman_cfg.get("process_noise", 1e-2),
            measurement_noise=kalman_cfg.get("measurement_noise", 1e-1),
        )
        self.occlusion_handler = OcclusionHandler(cfg)

        # Exporter is created per-run in run()
        self._exporter: Optional[SLPExporter] = None
        self._output_path: str = export_cfg.get("output_path", "output.slp")
        self._low_conf_threshold: float = export_cfg.get("low_confidence_threshold", 0.4)

    def run(self, video_path: str, output_path: Optional[str] = None) -> None:
        """Process an entire video file and write a .slp output."""
        out_path = output_path or self._output_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Processing {Path(video_path).name} — {total_frames} frames @ {fps:.1f} FPS")

        # Skeleton node names come from the SLEAP model
        skeleton_names = self.sleap._node_names

        self._exporter = SLPExporter(
            output_path=out_path,
            skeleton_names=skeleton_names,
            video_path=video_path,
            low_confidence_threshold=self._low_conf_threshold,
        )

        frame_id = 0
        with tqdm(total=total_frames, unit="frame", desc="SLEAP-Seg") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pose_results, reid_confs = self._process_frame(frame, frame_id)

                self._exporter.add_frame(frame_id, pose_results, reid_confs)

                frame_id += 1
                pbar.update(1)

        cap.release()
        self._exporter.flush()
        print(f"Done. {self._exporter.flagged_frame_count} frames flagged for review.")

    def _process_frame(self, frame: np.ndarray, frame_id: int):
        """Run one frame through the full pipeline. Returns (pose_results, reid_confs)."""
        # 1. Segmentation
        detections = self.seg_backend.detect(frame)

        # 2. Tracking (ByteTrack + Re-ID)
        active_tracks = self.tracker.update(frame, detections)

        if not active_tracks:
            return [], {}

        # Build a lookup from track_id to mask
        mask_by_id = {t.track_id: t.mask for t in active_tracks}

        # 3. SLEAP inference on each crop
        bboxes = [t.bbox for t in active_tracks]
        track_ids = [t.track_id for t in active_tracks]
        pose_results = self.sleap.infer(frame, bboxes, track_ids, frame_id)

        # 4. Occlusion detection & smoothing
        pose_results = self.occlusion_handler.process(pose_results, active_tracks, frame_id)

        # 5. Mask constraint + Kalman interpolation
        filtered_results = []
        for result in pose_results:
            mask = mask_by_id.get(result.track_id)
            result = self.kp_filter.filter(result, mask)
            filtered_results.append(result)

        # 6. Collect Re-ID confidences from the fused tracker
        reid_confs = {}
        for tid in track_ids:
            emb = self.tracker.reid.bank.get(tid)
            if emb is not None:
                # Use self-similarity as a proxy for track stability (1.0 = stable)
                reid_confs[tid] = float(self.tracker.reid.bank.similarity(emb, tid))

        return filtered_results, reid_confs
