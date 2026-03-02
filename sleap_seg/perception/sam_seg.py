"""SAM (Segment Anything) wrapper: refines coarse YOLO masks using bbox prompts."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .yolo_seg import Detection


class SAMRefiner:
    """Uses SAM to refine masks produced by YOLOv8-seg.

    Triggered selectively (when YOLO mask confidence is below a threshold) to
    avoid unnecessary latency on every frame.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: str = "cpu",
    ) -> None:
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self._current_frame: Optional[np.ndarray] = None

    def set_image(self, frame_rgb: np.ndarray) -> None:
        """Encode image features once per frame (amortised across all detections)."""
        self.predictor.set_image(frame_rgb)
        self._current_frame = frame_rgb

    def refine_detection(self, detection: Detection) -> Detection:
        """Refine a single Detection's mask using its bounding box as a prompt."""
        x1, y1, x2, y2 = detection.bbox
        input_box = np.array([x1, y1, x2, y2])

        masks, scores, _ = self.predictor.predict(
            box=input_box[None, :],
            multimask_output=True,
        )
        # Pick the mask with the highest IoU score
        best_idx = int(np.argmax(scores))
        refined_mask = masks[best_idx].astype(np.uint8)

        return Detection(
            bbox=detection.bbox,
            mask=refined_mask,
            confidence=float(scores[best_idx]),
            class_id=detection.class_id,
            track_id=detection.track_id,
        )

    def refine_detections(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        trigger_conf: float = 0.5,
    ) -> List[Detection]:
        """Refine any detections whose YOLO confidence is below trigger_conf.

        Only encodes the image once (lazy), skips SAM for high-confidence detections.
        """
        needs_refinement = [d.confidence < trigger_conf for d in detections]

        if not any(needs_refinement):
            return detections

        frame_rgb = frame_bgr[:, :, ::-1]
        self.set_image(frame_rgb)

        refined: List[Detection] = []
        for det, do_refine in zip(detections, needs_refinement):
            if do_refine:
                refined.append(self.refine_detection(det))
            else:
                refined.append(det)

        return refined
