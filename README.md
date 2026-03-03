# SLEAP-Seg

A segmentation-guided pose estimation pipeline for multi-animal tracking. Combines **YOLOv8-seg** instance segmentation with **ByteTrack + OSNet Re-ID** tracking and **SLEAP-NN** keypoint inference — eliminating ID switches and keypoint jitter in multi-animal behavioral analysis..

## Problem

Standard SLEAP inference on multiple visually similar animals (e.g., C57BL/6 mice) suffers from:
- **ID Switch**: Animals touching/separating causes identity confusion across frames
- **Occlusion Jitter**: Keypoints "fly" to background when animals overlap
- **Identity Loss**: After prolonged occlusion, animals may be permanently mislabeled

## Architecture

```
Video → [YOLOv8-seg] → [ByteTrack + OSNet Re-ID] → [SLEAP-NN] → .slp output
          Perception       Tracking Layer              Pose Layer
```

| Layer | Components | Function |
|-------|-----------|----------|
| Perception | YOLOv8-seg, SAM | Per-frame instance masks & bounding boxes |
| Tracking | ByteTrack (Mask-IoU), OSNet Re-ID | Stable ID across frames and occlusions |
| Pose | SLEAP-NN (.ckpt) or legacy SLEAP (.h5) | Keypoint inference with mask constraints |
| Post-proc | Kalman filter, linear interpolation | Smooth keypoints during occlusion |

---

## Installation

### Main Environment (sleapSeg — Python 3.10)

```bash
conda env create -f environment.yml
conda activate sleapSeg
pip install -e .
```

### SLEAP-NN Environment (sleapNN — Python 3.11)

Required if your SLEAP model is in `.ckpt` format (exported from SLEAP ≥ 1.4).

```bash
conda create -n sleapNN python=3.11 -y
conda activate sleapNN
pip install git+https://github.com/talmolab/sleap-nn.git torchvision
```

---

## Usage

### Full Pipeline (one command)

```bash
conda activate sleapSeg

sleap-seg run \
  --video my_video.mp4 \
  --seg-model models/yolov8_mice.pt \
  --sleap-model /path/to/model_dir \
  --output output.slp \
  --config config/default.yaml \
  --device mps
```

If your SLEAP model is a `.ckpt` file, the pipeline **automatically** calls the `sleapNN` environment to pre-compute keypoints before processing.

### Step-by-Step Workflow

```bash
# 1. Pre-compute SLEAP keypoints (Python 3.11, sleapNN env)
sleap-seg precompute \
  --video my_video.mp4 \
  --sleap-model /path/to/model_dir

# 2. Run main pipeline using cached keypoints
sleap-seg run \
  --video my_video.mp4 \
  --seg-model models/yolov8_mice.pt \
  --sleap-cache my_video_sleap_kps.npz \
  --output output.slp
```

### Validate Results

```bash
sleap-seg validate \
  --gt  ground_truth.slp \
  --pred output.slp \
  --baseline pure_sleap_output.slp \
  --report validation_report.csv
```

---

## Training a Custom YOLOv8 Model

The bundled generic YOLO model does not know your specific experimental setup.
Follow this workflow to train a custom model:

### Step 1: Extract frames for annotation

```bash
python scripts/extract_frames.py \
    --folder /path/to/videos/ \
    --target 200 \
    --output frames_to_label/
```

### Step 2: Annotate with LabelMe

See **[LABELING_GUIDE.md](LABELING_GUIDE.md)** for detailed instructions.

```bash
pip install "labelme[ai]"
labelme frames_to_label/
```

Use labels `mouse_1`, `mouse_2` (lower-case, underscore).
Aim for **≥ 30% of frames with occlusion**.

### Step 3: Train

```bash
python scripts/train_yolo.py \
    --labels frames_to_label/ \
    --classes mouse_1 mouse_2 \
    --epochs 100 \
    --device mps
```

The script auto-converts annotations, trains, and updates `config/default.yaml`.

---

## Configuration

```
config/
├── default.yaml        # Base configuration (override per-experiment)
├── lab_whitebox.yaml   # White box / bright lighting
└── lab_blackbox.yaml   # Dark box / dim lighting
```

Key parameters:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `perception` | `yolo_conf` | 0.25 | YOLO detection confidence threshold |
| `tracking` | `match_thresh` | 0.8 | ByteTrack IoU match threshold |
| `reid` | `cosine_threshold` | 0.7 | Min cosine similarity for Re-ID |
| `occlusion` | `iou_threshold` | 0.6 | Mask IoU that triggers occlusion mode |
| `pose` | `peak_threshold` | 0.2 | Min keypoint confidence |

Use a lab-specific config:
```bash
sleap-seg run --config config/lab_blackbox.yaml ...
```

---

## Project Structure

```
SLEAP-Seg/
├── environment.yml
├── setup.py
├── config/
│   ├── default.yaml
│   ├── lab_whitebox.yaml
│   └── lab_blackbox.yaml
├── sleap_seg/
│   ├── perception/     # YOLOv8-seg, SAM wrappers
│   ├── tracking/       # ByteTrack, OSNet Re-ID
│   ├── pose/           # SLEAP-NN / legacy SLEAP inferencer
│   ├── occlusion/      # Occlusion detection & interpolation
│   ├── export/         # .slp exporter
│   └── pipeline.py     # Main orchestrator
├── cli/
│   └── run.py          # CLI entry point
├── scripts/
│   ├── extract_frames.py    # Frame extraction for annotation
│   ├── train_yolo.py        # LabelMe → YOLO format + training
│   ├── sleap_nn_worker.py   # Batch SLEAP-NN inference worker (sleapNN env)
│   ├── validate.py          # Quantitative evaluation (IDS, RMSE)
│   └── visualize.py         # Debug overlay visualization
├── LABELING_GUIDE.md
└── data/
    └── models/
```

---

## Development Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | 🔄 In Progress | Custom YOLO training (requires user annotation) |
| Phase 2 | ✅ Done | SLEAP-NN env (Python 3.11) — model loads and infers |
| Phase 3 | ✅ Done | Validation script (ID switches + occlusion RMSE) |
| Phase 4 | ✅ Done | Lab config templates + one-click training + docs |

---

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| ID Switches / 10min video | < 5 | Needs benchmark |
| Occlusion RMSE reduction | ≥ 30% vs pure SLEAP | Needs benchmark |
| Processing speed (RTX 3090) | ≥ 30 FPS | Not yet tested on GPU |

---

## Requirements

- macOS (M1/M2 tested with `--device mps`) or Linux with CUDA GPU
- Python 3.10 (sleapSeg env) + Python 3.11 (sleapNN env)
- ~8 GB RAM, ~4 GB VRAM for GPU inference
- SLEAP model in `.ckpt` format (SLEAP ≥ 1.4) or `.h5` format (SLEAP ≤ 1.3)
