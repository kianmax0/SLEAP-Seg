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
@click.option("--video", "-v", required=True, type=click.Path(exists=True), help="Input video file (.mp4, .avi, ...)")
@click.option("--seg-model", required=True, type=click.Path(exists=True), help="Path to YOLOv8-seg model weights (.pt)")
@click.option("--sleap-model", required=True, type=click.Path(exists=True), help="Path to SLEAP model (.pkg.slp)")
@click.option("--output", "-o", default="output.slp", show_default=True, help="Output .slp file path")
@click.option("--config", "-c", default="config/default.yaml", show_default=True, type=click.Path(), help="Path to YAML config file")
@click.option("--device", default=None, help="Compute device: cpu | cuda:0 | mps (overrides config)")
@click.option("--sam-checkpoint", default=None, type=click.Path(), help="SAM checkpoint path (overrides config)")
@click.option("--seg-backend", default=None, type=click.Choice(["yolo", "sam", "yolo+sam"]), help="Segmentation backend (overrides config)")
def run(
    video: str,
    seg_model: str,
    sleap_model: str,
    output: str,
    config: str,
    device: Optional[str],
    sam_checkpoint: Optional[str],
    seg_backend: Optional[str],
) -> None:
    """Run the full SLEAP-Seg pipeline on a video file."""
    from sleap_seg.pipeline import Pipeline

    cfg = _load_config(config)

    # Apply CLI overrides
    cfg["perception"]["yolo_model"] = seg_model
    cfg["pose"]["sleap_model"] = sleap_model
    cfg["export"]["output_path"] = output

    if device:
        cfg["device"] = device
    if sam_checkpoint:
        cfg["perception"]["sam_checkpoint"] = sam_checkpoint
    if seg_backend:
        cfg["perception"]["seg_backend"] = seg_backend

    click.echo(f"Config: {config}")
    click.echo(f"Device: {cfg.get('device', 'cpu')}")
    click.echo(f"Seg backend: {cfg['perception']['seg_backend']}")
    click.echo(f"Output: {output}")

    pipeline = Pipeline(cfg)
    pipeline.run(video, output_path=output)


@main.command()
@click.option("--config", "-c", default="config/default.yaml", show_default=True, help="Config to validate")
def validate_config(config: str) -> None:
    """Validate a YAML config file and print a summary."""
    cfg = _load_config(config)
    click.echo(click.style("Config loaded successfully:", fg="green"))
    click.echo(yaml.dump(cfg, default_flow_style=False))


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
