"""OSNet-based Re-ID feature extractor and embedding bank.

Uses torchreid (deep-person-reid) with a pretrained OSNet model to extract
512-dim appearance embeddings from masked animal crops. Maintains a per-track
embedding bank updated via exponential moving average (EMA).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_OSNET_INPUT_SIZE = (256, 128)  # (H, W) standard Re-ID input


def _load_osnet(model_name: str, weights_path: Optional[str], device: str) -> torch.nn.Module:
    """Load pretrained OSNet from torchreid."""
    import torchreid

    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1000,    # placeholder; we only use the feature extractor
        pretrained=(weights_path is None),
    )

    if weights_path and Path(weights_path).exists():
        torchreid.utils.load_pretrained_weights(model, weights_path)

    model.eval()
    model.to(device)
    return model


def _preprocess_crop(
    frame: np.ndarray,
    mask: np.ndarray,
    bbox: np.ndarray,
) -> torch.Tensor:
    """Crop the masked animal region and convert to a normalised tensor."""
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    crop = frame[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2, None]  # (H, W, 1)

    # Zero-out background pixels
    crop = crop * mask_crop

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (_OSNET_INPUT_SIZE[1], _OSNET_INPUT_SIZE[0]))

    tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor  # (3, H, W)


class ReIDBank:
    """Per-track embedding storage with EMA update."""

    def __init__(self, ema_alpha: float = 0.9) -> None:
        self.alpha = ema_alpha
        self._embeddings: Dict[int, np.ndarray] = {}

    def update(self, track_id: int, embedding: np.ndarray) -> None:
        if track_id in self._embeddings:
            self._embeddings[track_id] = (
                self.alpha * self._embeddings[track_id]
                + (1.0 - self.alpha) * embedding
            )
        else:
            self._embeddings[track_id] = embedding.copy()

    def get(self, track_id: int) -> Optional[np.ndarray]:
        return self._embeddings.get(track_id)

    def similarity(self, embedding: np.ndarray, track_id: int) -> float:
        """Cosine similarity in [0, 1] between query and stored embedding."""
        stored = self.get(track_id)
        if stored is None:
            return 0.0
        dot = float(np.dot(embedding, stored))
        norm = float(np.linalg.norm(embedding) * np.linalg.norm(stored))
        return dot / norm if norm > 0 else 0.0

    def find_best_match(
        self,
        embedding: np.ndarray,
        candidate_ids: List[int],
        threshold: float = 0.7,
    ) -> Tuple[Optional[int], float]:
        """Return (best_track_id, similarity) or (None, 0) if no match above threshold."""
        best_id: Optional[int] = None
        best_sim = threshold - 1e-6

        for tid in candidate_ids:
            sim = self.similarity(embedding, tid)
            if sim > best_sim:
                best_sim = sim
                best_id = tid

        return best_id, best_sim

    def remove(self, track_id: int) -> None:
        self._embeddings.pop(track_id, None)


class ReIDExtractor:
    """Wraps an OSNet model and ReIDBank for embedding extraction and querying."""

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        weights_path: Optional[str] = None,
        embedding_dim: int = 512,
        ema_alpha: float = 0.9,
        cosine_threshold: float = 0.7,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.cosine_threshold = cosine_threshold
        self.bank = ReIDBank(ema_alpha=ema_alpha)

        self._model = _load_osnet(model_name, weights_path, device)

    @torch.no_grad()
    def extract(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        bboxes: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Extract L2-normalised embeddings for a batch of instances."""
        if not masks:
            return []

        tensors = [
            _preprocess_crop(frame, m, b)
            for m, b in zip(masks, bboxes)
        ]
        batch = torch.stack(tensors).to(self.device)

        features = self._model(batch)          # (N, embedding_dim)
        features = F.normalize(features, dim=1)

        return [features[i].cpu().numpy() for i in range(features.shape[0])]

    def update_bank(self, track_ids: List[int], embeddings: List[np.ndarray]) -> None:
        for tid, emb in zip(track_ids, embeddings):
            self.bank.update(tid, emb)

    def reassociate(
        self,
        lost_track_ids: List[int],
        query_embedding: np.ndarray,
    ) -> Tuple[Optional[int], float]:
        """Find the best matching lost track for a re-appearing embedding."""
        return self.bank.find_best_match(
            query_embedding,
            candidate_ids=lost_track_ids,
            threshold=self.cosine_threshold,
        )
