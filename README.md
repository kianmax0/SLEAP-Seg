# SLEAP-Seg

A segmentation-guided pose estimation pipeline for multi-animal tracking. Combines YOLOv8-seg + SAM instance segmentation with ByteTrack + OSNet Re-ID tracking to eliminate ID switches and keypoint jitter in SLEAP-based behavioral analysis.

## Problem

Standard SLEAP inference on multiple visually identical animals (e.g., C57BL/6 black mice) suffers from:
- **ID Switch**: Animals touching/separating causes identity confusion across frames
- **Occlusion Jitter**: Keypoints "fly" to background when animals overlap

## Architecture

```
Video → [YOLOv8-seg + SAM] → [ByteTrack + OSNet Re-ID] → [SLEAP per-crop] → .slp output
          Perception Layer        Tracking Layer              Pose Layer
```

- **Perception**: Per-frame instance masks and bounding boxes
- **Tracking**: Mask-IoU extended ByteTrack with OSNet appearance Re-ID for post-occlusion recovery
- **Pose**: SLEAP inference constrained by masks; Kalman filter interpolates unreliable keypoints
- **Occlusion**: Mask IoU > 0.6 triggers occlusion mode with linear keypoint smoothing

## Installation

```bash
conda env create -f environment.yml
conda activate sleapSeg
pip install -e .
```

> For CUDA (RTX 3090) deployment, replace the `torch` / `torchvision` lines in `environment.yml` with your CUDA-compatible wheels before creating the environment.

## Usage

```bash
sleap-seg run \
  --video input.mp4 \
  --seg-model models/yolov8n-seg.pt \
  --sleap-model models/model.pkg.slp \
  --output output.slp \
  --config config/default.yaml \
  --device cuda:0
```

## Project Structure

```
SLEAP-Seg/
├── environment.yml
├── setup.py
├── config/
│   └── default.yaml
├── sleap_seg/
│   ├── perception/      # YOLOv8-seg + SAM wrappers
│   ├── tracking/        # ByteTrack + OSNet Re-ID
│   ├── pose/            # SLEAP inference + Kalman constraint
│   ├── occlusion/       # Occlusion detection & smoothing
│   ├── export/          # .slp writer + low-confidence flagging
│   └── pipeline.py      # Main frame-loop orchestrator
├── cli/
│   └── run.py           # Click CLI entry point
├── scripts/
│   └── visualize.py     # Debug overlay visualization
├── data/                # Input videos (gitignored)
└── models/              # Model weights (gitignored)
```

## Performance Target

≥ 30 FPS on a single RTX 3090 (fp16 inference, batched YOLO + SLEAP).

## Success Metrics

| Metric | Target |
|--------|--------|
| ID switch reduction | ≥ 80% fewer switches in 10-min social video |
| Keypoint RMSE (50%+ occlusion) | ≥ 30% reduction |
| Manual correction time (1h video) | 2 hours → 30 minutes |
