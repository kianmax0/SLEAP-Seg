"""Factory that builds the configured segmentation backend."""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np

from .yolo_seg import Detection, YOLOSegDetector
from .sam_seg import SAMRefiner


class SegmentationBackend:
    """Unified interface for segmentation.

    Supports three modes controlled by ``seg_backend`` config key:
    - ``"yolo"``      – YOLOv8-seg only
    - ``"yolo+sam"``  – YOLO coarse detection, SAM mask refinement on low-conf detections
    - ``"sam"``       – SAM with manual bbox prompts (not recommended for automated pipelines)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        perception_cfg = cfg.get("perception", {})
        self.backend_mode: str = perception_cfg.get("seg_backend", "yolo")
        device: str = cfg.get("device", "cpu")

        self.yolo: Optional[YOLOSegDetector] = None
        self.sam: Optional[SAMRefiner] = None

        if "yolo" in self.backend_mode:
            self.yolo = YOLOSegDetector(
                model_path=perception_cfg["yolo_model"],
                imgsz=perception_cfg.get("yolo_imgsz", 640),
                conf=perception_cfg.get("yolo_conf", 0.25),
                fp16=perception_cfg.get("yolo_fp16", False),
                device=device,
            )

        if "sam" in self.backend_mode:
            self.sam = SAMRefiner(
                checkpoint_path=perception_cfg["sam_checkpoint"],
                model_type=perception_cfg.get("sam_model_type", "vit_h"),
                device=device,
            )

        self.sam_trigger_conf: float = perception_cfg.get("sam_trigger_conf", 0.5)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect all instances in a single BGR frame."""
        if self.yolo is None:
            raise RuntimeError("No YOLO model loaded; cannot run detection.")

        detections = self.yolo.detect(frame)

        if self.sam is not None and detections:
            detections = self.sam.refine_detections(
                frame, detections, trigger_conf=self.sam_trigger_conf
            )

        return detections


def build_backend(cfg: Dict[str, Any]) -> SegmentationBackend:
    """Construct a SegmentationBackend from a config dictionary."""
    return SegmentationBackend(cfg)
