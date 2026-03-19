"""Convert LabelMe JSON annotations to YOLO format and train YOLOv8-seg.

Usage:

  # Convert + train in one step
  python scripts/train_yolo.py \\
      --labels data/frames_to_label/ \\
      --output runs/mice_seg/ \\
      --base-model models/yolov8n-seg.pt \\
      --epochs 100

  # Only convert (skip training)
  python scripts/train_yolo.py --labels data/frames_to_label/ --output dataset/ --convert-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────
#  LabelMe → YOLO conversion
# ─────────────────────────────────────────────────────────────────

def _labelme_shape_to_yolo_polygon(
    shape: dict, img_w: int, img_h: int
) -> Optional[List[float]]:
    """Convert a single LabelMe polygon shape to normalised YOLO flat list."""
    if shape["shape_type"] not in ("polygon", "rectangle"):
        return None
    pts = shape["points"]
    if shape["shape_type"] == "rectangle":
        (x1, y1), (x2, y2) = pts[0], pts[1]
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if len(pts) < 3:
        return None
    flat: List[float] = []
    for x, y in pts:
        flat.extend([x / img_w, y / img_h])
    return flat


def convert_labelme_to_yolo(
    labels_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.1,
    class_names: Optional[List[str]] = None,
) -> Path:
    """Convert a directory of LabelMe JSON files to YOLO segmentation format.

    Returns the path to the generated dataset.yaml.
    """
    json_files = sorted(labels_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {labels_dir}")
        sys.exit(1)

    # Discover classes if not specified (default is ["mouse"])
    if class_names is None:
        found: set = set()
        for jf in json_files:
            data = json.loads(jf.read_text())
            for s in data.get("shapes", []):
                found.add(s["label"])
        class_names = sorted(found)
        print(f"Discovered {len(class_names)} classes: {class_names}")
    else:
        print(f"Using specified classes: {class_names}")

    class_to_id = {c: i for i, c in enumerate(class_names)}

    # Create output dirs
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Split train/val
    import random
    random.seed(42)
    shuffled = list(json_files)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_set = set(str(f) for f in shuffled[:n_val])

    stats = {"train": 0, "val": 0, "skipped": 0}

    for jf in json_files:
        data = json.loads(jf.read_text())
        img_filename = data.get("imagePath", jf.stem + ".jpg")
        img_path = labels_dir / img_filename

        # Try alternative extensions if exact path not found
        if not img_path.exists():
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                candidate = labels_dir / (jf.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

        split = "val" if str(jf) in val_set else "train"
        dest_img = output_dir / "images" / split / img_path.name
        dest_lbl = output_dir / "labels" / split / (img_path.stem + ".txt")

        # Get image dimensions
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if img_w is None or img_h is None:
            if img_path.exists():
                import cv2
                img = cv2.imread(str(img_path))
                img_h, img_w = img.shape[:2]
            else:
                stats["skipped"] += 1
                continue

        # Write label file
        lines: List[str] = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            if label not in class_to_id:
                continue
            cls_id = class_to_id[label]
            poly = _labelme_shape_to_yolo_polygon(shape, img_w, img_h)
            if poly is None:
                continue
            coords_str = " ".join(f"{v:.6f}" for v in poly)
            lines.append(f"{cls_id} {coords_str}")

        if not lines:
            stats["skipped"] += 1
            continue

        dest_lbl.write_text("\n".join(lines))

        if img_path.exists():
            shutil.copy2(img_path, dest_img)

        stats[split] += 1

    print(f"Converted: {stats['train']} train, {stats['val']} val, {stats['skipped']} skipped")

    # Write dataset.yaml
    dataset_yaml = output_dir / "dataset.yaml"
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    dataset_yaml.write_text(yaml.dump(yaml_content, default_flow_style=False))
    print(f"Dataset YAML written to: {dataset_yaml}")
    return dataset_yaml


# ─────────────────────────────────────────────────────────────────
#  YOLO training
# ─────────────────────────────────────────────────────────────────

def train_yolo(
    dataset_yaml: Path,
    base_model: str,
    output_dir: Path,
    epochs: int,
    imgsz: int,
    device: str,
    run_name: str,
) -> Path:
    """Run YOLOv8-seg fine-tuning. Returns path to best weights."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(base_model)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        name=run_name,
        project=str(output_dir),
        exist_ok=True,
        verbose=True,
    )

    best_weights = output_dir / run_name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    return best_weights


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe annotations to YOLO format and train YOLOv8-seg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--labels", type=Path, required=True,
                        help="Directory with LabelMe JSON files (and images)")
    parser.add_argument("--output", "-o", type=Path, default=Path("runs/mice_seg"),
                        help="Output directory for dataset and training results")
    parser.add_argument("--base-model", default="models/yolov8n-seg.pt",
                        help="Base YOLO model to fine-tune (default: yolov8n-seg.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="",
                        help="Device: '' (auto), 'cpu', 'mps', '0' (GPU 0)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of images for validation (default: 0.1)")
    parser.add_argument("--run-name", default="mice_fvb_seg",
                        help="Name of the training run folder")
    parser.add_argument("--classes", nargs="+", default=["mouse"],
                        help="Class names in label order. Default: ['mouse']. "
                             "All mice use the same label regardless of identity — "
                             "individual IDs are assigned by the ByteTrack/Re-ID layer at inference time.")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert annotations; skip training.")
    parser.add_argument("--copy-model", type=Path, default=Path("models/yolov8_mice.pt"),
                        help="After training, copy best.pt here and update config. Default: models/yolov8_mice.pt")
    parser.add_argument("--update-config", action="store_true", default=True,
                        help="Update config/default.yaml with the new model path after training.")

    args = parser.parse_args()

    dataset_dir = args.output / "dataset"
    dataset_yaml = convert_labelme_to_yolo(
        labels_dir=args.labels,
        output_dir=dataset_dir,
        val_fraction=args.val_split,
        class_names=args.classes,
    )

    if args.convert_only:
        print("--convert-only specified. Skipping training.")
        return

    best_weights = train_yolo(
        dataset_yaml=dataset_yaml,
        base_model=args.base_model,
        output_dir=args.output,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        run_name=args.run_name,
    )

    # Copy best model to models/
    if best_weights.exists():
        args.copy_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, args.copy_model)
        print(f"Best model copied to: {args.copy_model}")

        if args.update_config:
            config_path = Path("config/default.yaml")
            if config_path.exists():
                text = config_path.read_text()
                # Replace yolo_model path
                import re
                new_text = re.sub(
                    r"(yolo_model\s*:\s*).*",
                    f"\\1{args.copy_model}",
                    text,
                )
                config_path.write_text(new_text)
                print(f"Updated config/default.yaml: yolo_model → {args.copy_model}")

    print("\nAll done! You can now run the visualizer to verify:")
    print(f"  python scripts/visualize.py --video YOUR_VIDEO.mp4 --skip-sleap")


if __name__ == "__main__":
    main()
