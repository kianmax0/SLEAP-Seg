"""Export pose results to SLEAP-native .slp (HDF5) format.

Writes a sleap.Labels object containing:
- All frames with per-track Instance predictions
- A 'suggestions' list flagging low-confidence frames for human review
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..pose.sleap_infer import PoseResult


class SLPExporter:
    """Accumulates per-frame PoseResults and writes a .slp file on flush()."""

    def __init__(
        self,
        output_path: str,
        skeleton_names: List[str],
        video_path: str,
        low_confidence_threshold: float = 0.4,
    ) -> None:
        self.output_path = Path(output_path)
        self.skeleton_names = skeleton_names
        self.video_path = video_path
        self.low_confidence_threshold = low_confidence_threshold

        # { frame_id: List[PoseResult] }
        self._frame_buffer: Dict[int, List[PoseResult]] = {}
        # Frame IDs flagged for human review
        self._flagged_frames: List[int] = []

    def add_frame(
        self,
        frame_id: int,
        pose_results: List[PoseResult],
        reid_confidences: Optional[Dict[int, float]] = None,
    ) -> None:
        """Buffer pose results for one frame; flag if Re-ID confidence is low."""
        self._frame_buffer[frame_id] = pose_results

        if reid_confidences:
            min_conf = min(reid_confidences.values()) if reid_confidences else 1.0
            if min_conf < self.low_confidence_threshold:
                self._flagged_frames.append(frame_id)
        # Also flag frames with many untrusted keypoints
        for result in pose_results:
            trusted_ratio = sum(kp.trusted for kp in result.keypoints) / max(
                1, len(result.keypoints)
            )
            if trusted_ratio < self.low_confidence_threshold:
                if frame_id not in self._flagged_frames:
                    self._flagged_frames.append(frame_id)

    def flush(self) -> None:
        """Write accumulated results to a .slp file."""
        import sleap

        video = sleap.Video.from_filename(self.video_path)
        skeleton = sleap.Skeleton(name="Mouse")
        for name in self.skeleton_names:
            skeleton.add_node(name)

        labeled_frames: List[sleap.LabeledFrame] = []

        for frame_id in sorted(self._frame_buffer.keys()):
            pose_results = self._frame_buffer[frame_id]
            instances: List[sleap.PredictedInstance] = []

            for result in pose_results:
                points_dict = {}
                for kp in result.keypoints:
                    points_dict[kp.name] = sleap.PredictedPoint(
                        x=kp.x if not np.isnan(kp.x) else 0.0,
                        y=kp.y if not np.isnan(kp.y) else 0.0,
                        score=kp.score,
                        visible=kp.trusted and not (np.isnan(kp.x) or np.isnan(kp.y)),
                    )

                instance = sleap.PredictedInstance.from_numpy(
                    points=np.array([[kp.x, kp.y] for kp in result.keypoints]),
                    point_scores=np.array([kp.score for kp in result.keypoints]),
                    instance_score=float(
                        np.nanmean([kp.score for kp in result.keypoints])
                    ),
                    skeleton=skeleton,
                    track=sleap.Track(spawned_on=frame_id, name=f"mouse_{result.track_id}"),
                )
                instances.append(instance)

            lf = sleap.LabeledFrame(video=video, frame_idx=frame_id, instances=instances)
            labeled_frames.append(lf)

        # Build suggestions list for flagged frames
        suggestions = [
            sleap.SuggestionFrame(video=video, frame_idx=fid)
            for fid in sorted(set(self._flagged_frames))
        ]

        labels = sleap.Labels(
            labeled_frames=labeled_frames,
            videos=[video],
            skeletons=[skeleton],
            suggestions=suggestions,
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sleap.Labels.save_file(labels, str(self.output_path))
        print(
            f"Saved {len(labeled_frames)} frames to {self.output_path} "
            f"({len(suggestions)} frames flagged for review)"
        )

    @property
    def flagged_frame_count(self) -> int:
        return len(set(self._flagged_frames))
