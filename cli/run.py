"""Click-based CLI entry point for SLEAP-Seg."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import yaml


@click.group()
def main():
    """SLEAP-Seg: segmentation-guided pose estimation for multi-animal tracking."""


@main.command()
@click.option("--video", "-v", required=True, type=click.Path(exists=True),
              help="Input video file (.mp4, .avi, ...)")
@click.option("--seg-model", required=True, type=click.Path(exists=True),
              help="Path to YOLOv8-seg model weights (.pt)")
@click.option("--sleap-model", default=None, type=click.Path(),
              help="Path to SLEAP model directory (containing best.ckpt or best_model.h5)")
@click.option("--sleap-cache", default=None, type=click.Path(),
              help="Pre-computed SLEAP keypoints .npz file. If not provided and sleapNN env "
                   "is available, keypoints will be computed automatically.")
@click.option("--output", "-o", default="output.slp", show_default=True,
              help="Output .slp file path")
@click.option("--config", "-c", default="config/default.yaml", show_default=True,
              type=click.Path(), help="Path to YAML config file")
@click.option("--device", default=None,
              help="Compute device: cpu | cuda:0 | mps (overrides config)")
@click.option("--sam-checkpoint", default=None, type=click.Path(),
              help="SAM checkpoint path (overrides config)")
@click.option("--seg-backend", default=None,
              type=click.Choice(["yolo", "sam", "yolo+sam"]),
              help="Segmentation backend (overrides config)")
@click.option("--auto-precompute/--no-auto-precompute", default=True,
              help="Auto-run sleap_nn_worker in sleapNN env if cache is missing. Default: enabled")
def run(
    video: str,
    seg_model: str,
    sleap_model: Optional[str],
    sleap_cache: Optional[str],
    output: str,
    config: str,
    device: Optional[str],
    sam_checkpoint: Optional[str],
    seg_backend: Optional[str],
    auto_precompute: bool,
) -> None:
    """Run the full SLEAP-Seg pipeline on a video file.

    \b
    Quick-start example:
      sleap-seg run \\
        --video my_video.mp4 \\
        --seg-model models/yolov8_mice.pt \\
        --sleap-model /path/to/model_dir \\
        --output result.slp

    \b
    With pre-computed SLEAP cache (faster on repeated runs):
      # Step 1: compute keypoints in sleapNN env
      conda run -n sleapNN python scripts/sleap_nn_worker.py \\
          --model /path/to/model_dir --video my_video.mp4

      # Step 2: run main pipeline with cache
      sleap-seg run \\
        --video my_video.mp4 \\
        --seg-model models/yolov8_mice.pt \\
        --sleap-cache my_video_sleap_kps.npz \\
        --output result.slp
    """
    from sleap_seg.pipeline import Pipeline
    from sleap_seg.pose.sleap_infer import precompute_sleap_cache

    cfg = _load_config(config)

    # Apply CLI overrides
    cfg["perception"]["yolo_model"] = seg_model

    if sleap_model:
        cfg["pose"]["sleap_model"] = sleap_model

    cfg["export"]["output_path"] = output

    if device:
        cfg["device"] = device
    if sam_checkpoint:
        cfg["perception"]["sam_checkpoint"] = sam_checkpoint
    if seg_backend:
        cfg["perception"]["seg_backend"] = seg_backend

    # Auto-precompute SLEAP cache if needed
    effective_cache = sleap_cache
    if effective_cache is None and auto_precompute and cfg.get("pose", {}).get("sleap_model"):
        model_path = cfg["pose"]["sleap_model"]
        model_dir = Path(model_path)
        if (model_dir / "best.ckpt").exists():
            click.echo(click.style(
                "→ Detected .ckpt model. Attempting to pre-compute SLEAP keypoints "
                "using sleapNN conda env...", fg="yellow"
            ))
            cache = precompute_sleap_cache(
                video_path=video,
                model_path=model_path,
                device=cfg.get("device", "cpu"),
                conda_env="sleapNN",
            )
            if cache:
                effective_cache = cache
                click.echo(click.style(f"✓ SLEAP cache ready: {cache}", fg="green"))
            else:
                click.echo(click.style(
                    "  Could not pre-compute SLEAP cache. Keypoints will be NaN.\n"
                    "  Run: conda run -n sleapNN python scripts/sleap_nn_worker.py "
                    f"--model {model_path} --video {video}",
                    fg="red"
                ))

    click.echo(f"Config:       {config}")
    click.echo(f"Device:       {cfg.get('device', 'cpu')}")
    click.echo(f"Seg backend:  {cfg['perception']['seg_backend']}")
    click.echo(f"SLEAP cache:  {effective_cache or '(none)'}")
    click.echo(f"Output:       {output}")

    if effective_cache:
        cfg["pose"]["sleap_cache"] = effective_cache

    pipeline = Pipeline(cfg)
    pipeline.run(video, output_path=output)


@main.command()
@click.option("--video", "-v", required=True, type=click.Path(exists=True),
              help="Input video file")
@click.option("--sleap-model", required=True, type=click.Path(exists=True),
              help="Path to SLEAP model directory")
@click.option("--output", "-o", default=None,
              help="Output .npz path (default: <video>_sleap_kps.npz next to video)")
@click.option("--device", default="cpu", show_default=True,
              help="Device: cpu | mps | cuda")
def precompute(video: str, sleap_model: str, output: Optional[str], device: str) -> None:
    """Pre-compute SLEAP keypoints for a video using the sleapNN conda environment.

    This must be run ONCE per video before using the main `run` command when your
    SLEAP model is in .ckpt format (requires Python 3.11 via the sleapNN conda env).

    \b
    Example:
      sleap-seg precompute \\
        --video my_video.mp4 \\
        --sleap-model /path/to/model_dir

      # Then use the cache:
      sleap-seg run --video my_video.mp4 ... --sleap-cache my_video_sleap_kps.npz
    """
    from sleap_seg.pose.sleap_infer import precompute_sleap_cache

    result = precompute_sleap_cache(
        video_path=video,
        model_path=sleap_model,
        output_path=output,
        device=device,
        conda_env="sleapNN",
    )
    if result:
        click.echo(click.style(f"✓ Keypoints saved to: {result}", fg="green"))
    else:
        click.echo(click.style("✗ Pre-computation failed.", fg="red"))
        raise SystemExit(1)


@main.command()
@click.option("--config", "-c", default="config/default.yaml", show_default=True,
              help="Config to validate")
def validate_config(config: str) -> None:
    """Validate a YAML config file and print a summary."""
    cfg = _load_config(config)
    click.echo(click.style("Config loaded successfully:", fg="green"))
    click.echo(yaml.dump(cfg, default_flow_style=False))


@main.command()
@click.option("--gt", required=True, type=click.Path(exists=True),
              help="Ground-truth .slp file")
@click.option("--pred", required=True, type=click.Path(exists=True),
              help="Predicted .slp file (SLEAP-Seg output)")
@click.option("--baseline", default=None, type=click.Path(exists=True),
              help="Optional baseline .slp file (e.g. pure SLEAP output) for comparison")
@click.option("--iou-threshold", default=0.15, show_default=True,
              help="Bbox IoU threshold to classify occlusion frames")
@click.option("--report", default=None, type=click.Path(),
              help="Save results to CSV")
def validate(gt: str, pred: str, baseline: Optional[str],
             iou_threshold: float, report: Optional[str]) -> None:
    """Compute ID-switch and keypoint RMSE metrics against ground-truth labels."""
    import subprocess, sys
    cmd = [sys.executable, "scripts/validate.py",
           "--gt", gt, "--pred", pred,
           "--iou-threshold", str(iou_threshold)]
    if baseline:
        cmd += ["--baseline", baseline]
    if report:
        cmd += ["--report", report]
    subprocess.run(cmd, check=True)


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
