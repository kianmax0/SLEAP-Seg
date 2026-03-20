"""Main pipeline orchestrator: frame-loop connecting all modules (V2 state-aware)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .export.slp_exporter import SLPExporter
from .occlusion.occlusion_handler import OcclusionHandler
from .pose.keypoint_filter import KeypointFilter
from .pose.sleap_infer import PoseResult, SLEAPInferencer
from .perception.seg_backend import build_backend
from .state.frame_state import FrameState, build_pose_track_views, compute_frame_state
from .tracking.bytetrack import Track
from .tracking.tracker import FusedTracker


class Pipeline:
    """End-to-end SLEAP-Seg pipeline.

    Orchestrates:
      Perception → Tracking → FrameState → PoseTrack expansion →
      SLEAP inference (spatial/temporal assignment) → OcclusionHandler →
      Mask (optional) + Kalman → Export
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        device = cfg.get("device", "cpu")
        pose_cfg = cfg.get("pose", {})
        export_cfg = cfg.get("export", {})
        kalman_cfg = cfg.get("kalman", {})
        pipeline_cfg = cfg.get("pipeline", {})

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

        self._expected_mice: int = int(pipeline_cfg.get("expected_mice", 2))
        self._use_temporal_in_occlusion: bool = bool(
            pipeline_cfg.get("assignment", {}).get("use_temporal_in_occlusion", True)
        )
        self._disable_mask_when_risky: bool = bool(
            pipeline_cfg.get("occlusion", {}).get(
                "disable_mask_constraint_when_risky", True
            )
        )

        self._sticky_track_ids: List[int] = []
        self._prev_poses: Dict[int, PoseResult] = {}
        self._risk_streak: int = 0
        self._last_frame_state: Optional[FrameState] = None
        self._last_active_tracks: List[Track] = []

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

    def _process_frame(
        self, frame: Any, frame_id: int
    ) -> Tuple[List[PoseResult], Dict[int, float]]:
        """Run one frame through the full V2 pipeline."""
        detections = self.seg_backend.detect(frame)
        active_tracks = self.tracker.update(frame, detections)

        frame_state = compute_frame_state(detections, active_tracks, self.cfg)
        self._last_frame_state = frame_state

        if (
            frame_state == FrameState.PEACE
            and len(active_tracks) >= self._expected_mice
        ):
            self._sticky_track_ids = sorted(t.track_id for t in active_tracks)[
                : self._expected_mice
            ]

        if frame_state == FrameState.PEACE:
            self._risk_streak = 0
        else:
            self._risk_streak += 1

        if not active_tracks:
            self._prev_poses = {}
            self._last_active_tracks = []
            return [], {}

        self._last_active_tracks = list(active_tracks)

        views = build_pose_track_views(
            active_tracks,
            frame_state,
            self._sticky_track_ids,
            self._expected_mice,
        )
        bboxes = [v.bbox for v in views]
        track_ids = [v.track_id for v in views]

        use_temporal = (
            self._use_temporal_in_occlusion
            and frame_state
            in (FrameState.MERGED_BLOB, FrameState.PAIRWISE_OCCLUSION)
            and bool(self._prev_poses)
        )
        assignment_mode = "temporal" if use_temporal else "spatial"

        pose_results = self.sleap.infer(
            frame,
            bboxes,
            track_ids,
            frame_id,
            assignment_mode=assignment_mode,
            prev_poses=self._prev_poses if use_temporal else None,
        )

        pose_results = self.occlusion_handler.process(
            pose_results,
            active_tracks,
            frame_id,
            frame_state=frame_state,
            risk_streak=self._risk_streak,
        )

        risky = frame_state in (
            FrameState.MERGED_BLOB,
            FrameState.PAIRWISE_OCCLUSION,
        )
        apply_mask = not (self._disable_mask_when_risky and risky)

        mask_by_id = {t.track_id: t.mask for t in active_tracks}
        for v in views:
            if v.track_id not in mask_by_id:
                mask_by_id[v.track_id] = v.mask

        filtered_results: List[PoseResult] = []
        for result in pose_results:
            mask = mask_by_id.get(result.track_id)
            result = self.kp_filter.filter(result, mask, apply_mask=apply_mask)
            filtered_results.append(result)

        self._prev_poses = {r.track_id: r for r in filtered_results}

        reid_confs: Dict[int, float] = {}
        if self.tracker.reid is not None:
            for tid in track_ids:
                emb = self.tracker.reid.bank.get(tid)
                if emb is not None:
                    reid_confs[tid] = float(self.tracker.reid.bank.similarity(emb, tid))

        return filtered_results, reid_confs

    def process_frame(
        self, frame: Any, frame_id: int
    ) -> Tuple[List[PoseResult], Dict[int, float]]:
        """Public API for single-frame processing (e.g. visualize script)."""
        return self._process_frame(frame, frame_id)

    @property
    def last_frame_state(self) -> Optional[FrameState]:
        """Last computed FrameState (PEACE / MERGED_BLOB / PAIRWISE_OCCLUSION)."""
        return self._last_frame_state

    @property
    def last_active_tracks(self) -> List[Track]:
        """ByteTrack output for the last processed frame (before ghost expansion)."""
        return self._last_active_tracks
