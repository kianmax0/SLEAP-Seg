"""YOLOv8-seg wrapper: per-frame instance segmentation producing masks and bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    """Single instance detection from segmentation model."""

    bbox: np.ndarray          # [x1, y1, x2, y2] float32
    mask: np.ndarray          # Binary mask (H, W) uint8
    confidence: float
    class_id: int
    track_id: Optional[int] = None


class YOLOSegDetector:
    """Wraps ultralytics YOLOv8-seg for instance segmentation inference."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        fp16: bool = False,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.fp16 = fp16
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame. Returns list of Detection objects."""
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            half=self.fp16,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        result = results[0]

        if result.masks is None:
            return detections

        h, w = frame.shape[:2]
        masks_data = result.masks.data.cpu().numpy()   # (N, H', W')
        boxes_data = result.boxes

        for i in range(len(boxes_data)):
            xyxy = boxes_data[i].xyxy[0].cpu().numpy().astype(np.float32)
            conf_val = float(boxes_data[i].conf[0].cpu().numpy())
            cls_id = int(boxes_data[i].cls[0].cpu().numpy())

            # Resize mask back to original frame resolution
            raw_mask = masks_data[i]
            mask_resized = cv2.resize(
                raw_mask, (w, h), interpolation=cv2.INTER_NEAREST
            )
            binary_mask = (mask_resized > 0.5).astype(np.uint8)

            detections.append(
                Detection(
                    bbox=xyxy,
                    mask=binary_mask,
                    confidence=conf_val,
                    class_id=cls_id,
                )
            )

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run batched inference on multiple frames."""
        results = self.model(
            frames,
            imgsz=self.imgsz,
            conf=self.conf,
            half=self.fp16,
            device=self.device,
            verbose=False,
        )

        batch_detections: List[List[Detection]] = []
        for idx, result in enumerate(results):
            h, w = frames[idx].shape[:2]
            detections: List[Detection] = []

            if result.masks is None:
                batch_detections.append(detections)
                continue

            masks_data = result.masks.data.cpu().numpy()
            boxes_data = result.boxes

            for i in range(len(boxes_data)):
                xyxy = boxes_data[i].xyxy[0].cpu().numpy().astype(np.float32)
                conf_val = float(boxes_data[i].conf[0].cpu().numpy())
                cls_id = int(boxes_data[i].cls[0].cpu().numpy())

                raw_mask = masks_data[i]
                mask_resized = cv2.resize(
                    raw_mask, (w, h), interpolation=cv2.INTER_NEAREST
                )
                binary_mask = (mask_resized > 0.5).astype(np.uint8)

                detections.append(
                    Detection(
                        bbox=xyxy,
                        mask=binary_mask,
                        confidence=conf_val,
                        class_id=cls_id,
                    )
                )

            batch_detections.append(detections)

        return batch_detections
