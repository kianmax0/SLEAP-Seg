"""Extract frames from video files for annotation.

Usage examples:

  # Extract every 60 frames from one video
  python scripts/extract_frames.py --video input.mp4 --interval 60 --output frames/

  # Extract from all videos in a folder, targeting ~200 frames total
  python scripts/extract_frames.py --folder /path/to/videos/ --target 200 --output frames/

  # Use CLAHE-enhanced versions if available (recommended for dark videos)
  python scripts/extract_frames.py --folder /path/to/videos/ --prefer-clahe --target 200 --output frames/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}


def get_video_files(folder: Path, prefer_clahe: bool = False) -> List[Path]:
    """Return sorted list of video files from a folder.

    If prefer_clahe is True, prefer *_clahe.mp4 versions over the raw .Avi
    when both exist for the same recording session.
    """
    all_videos = sorted(
        p for p in folder.iterdir() if p.suffix in SUPPORTED_EXTENSIONS
    )

    if not prefer_clahe:
        return all_videos

    # Build a map: stem_without_clahe -> [paths]
    from collections import defaultdict
    groups: defaultdict = defaultdict(list)
    for v in all_videos:
        stem = v.stem.replace("_clahe", "")
        groups[stem].append(v)

    selected: List[Path] = []
    for stem, paths in sorted(groups.items()):
        clahe_paths = [p for p in paths if "clahe" in p.stem]
        selected.append(clahe_paths[0] if clahe_paths else paths[0])

    return selected


def extract_from_video(
    video_path: Path,
    output_dir: Path,
    interval: int,
    max_frames: Optional[int],
    prefix: str = "",
) -> int:
    """Extract frames from a single video at the given interval.

    Returns the number of frames saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARNING] Cannot open {video_path.name}, skipping.")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_s = total / fps if fps > 0 else 0

    print(f"  {video_path.name}: {total} frames @ {fps:.1f} FPS ({duration_s:.0f}s)")

    saved = 0
    frame_idx = 0
    pbar = tqdm(total=total, unit="frame", leave=False, desc=f"  Scanning")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            name = f"{prefix}{video_path.stem}_f{frame_idx:06d}.jpg"
            out_path = output_dir / name
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

            if max_frames and saved >= max_frames:
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"  Saved {saved} frames.")
    return saved


def compute_interval(video_paths: List[Path], target_total: int) -> int:
    """Compute a uniform interval to yield approximately target_total frames."""
    total_frames = 0
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if cap.isOpened():
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
    if total_frames == 0:
        return 60
    interval = max(1, total_frames // target_total)
    return interval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from video(s) for LabelMe annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Single video file path")
    src.add_argument("--folder", type=Path, help="Folder containing multiple videos")

    parser.add_argument(
        "--output", "-o", type=Path, default=Path("frames_to_label"),
        help="Output directory for extracted frames (default: frames_to_label/)"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=None,
        help="Extract every N-th frame. If omitted, computed automatically from --target."
    )
    parser.add_argument(
        "--target", "-t", type=int, default=200,
        help="Target number of frames to extract in total (used when --interval is not set). Default: 200"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Hard limit on total frames extracted across all videos."
    )
    parser.add_argument(
        "--prefer-clahe", action="store_true",
        help="When folder contains both raw and *_clahe versions, prefer the enhanced version."
    )
    parser.add_argument(
        "--include-occluded", action="store_true",
        help="Print reminder to include frames with mouse-on-mouse contact (for better annotation coverage)."
    )

    args = parser.parse_args()

    # Collect video paths
    if args.video:
        if not args.video.exists():
            print(f"Error: video not found: {args.video}")
            sys.exit(1)
        video_paths = [args.video]
    else:
        if not args.folder.exists():
            print(f"Error: folder not found: {args.folder}")
            sys.exit(1)
        video_paths = get_video_files(args.folder, prefer_clahe=args.prefer_clahe)
        if not video_paths:
            print(f"No video files found in {args.folder}")
            sys.exit(1)

    print(f"Found {len(video_paths)} video(s):")
    for vp in video_paths:
        print(f"  - {vp.name}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Determine interval
    interval = args.interval or compute_interval(video_paths, args.target)
    print(f"\nFrame extraction interval: every {interval} frames (~{args.target} total target)\n")

    # Extract
    total_saved = 0
    for vp in video_paths:
        remaining = None
        if args.max_frames:
            remaining = args.max_frames - total_saved
            if remaining <= 0:
                print("Reached max-frames limit, stopping.")
                break
        saved = extract_from_video(vp, args.output, interval, remaining)
        total_saved += saved

    print(f"\nDone. Extracted {total_saved} frames to: {args.output.resolve()}")
    print(f"Next step: labelme {args.output}/")

    if args.include_occluded:
        print(
            "\n[REMINDER] Make sure at least 30% of your labeled frames show two mice"
            " in close contact or overlapping. This is critical for training the"
            " model to handle occlusion correctly."
        )


if __name__ == "__main__":
    main()
