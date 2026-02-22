#!/usr/bin/env python3
"""
Load a dataset .npz produced by policy_to_dataset.py and write a video with
reward, return, and terminal status overlaid on each frame.

Usage:
  python visualize_dataset.py dataset.npz --output out.mp4
  python visualize_dataset.py dataset.npz --cap-length 10 --output short.mp4
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int = 14):
    """Return a font readable on Atari-sized frames (e.g. 210x160)."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except OSError:
            return ImageFont.load_default()


# Output size 210*4 x 160*4; text drawn at this resolution so it stays sharp
OUT_W, OUT_H = 160 * 4, 210 * 4


def draw_overlay(
    frame: np.ndarray,
    reward: float,
    terminal: bool,
    return_val: Optional[float] = None,
) -> np.ndarray:
    """Upscale frame to OUT_W x OUT_H, then draw overlay so text is high-res. Returns uint8 RGB."""
    img = Image.fromarray(frame).resize((OUT_W, OUT_H), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    font = _get_font(48)  # Large enough to be crisp at 4x resolution
    x, y = 16, 16
    line_height = 56
    green = (0, 255, 0)
    draw.text((x, y), f"Reward: {reward}", fill=green, font=font)
    y += line_height
    if return_val is not None:
        draw.text((x, y), f"Return: {return_val:.2f}", fill=green, font=font)
        y += line_height
    if terminal:
        draw.text((x, y), "Terminal: True", fill=green, font=font)
    else:
        draw.text((x, y), "Terminal: False", fill=green, font=font)
    return np.array(img)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a policy dataset as video with reward and terminal overlay.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the .npz dataset file",
    )
    parser.add_argument(
        "--cap-length",
        type=float,
        default=None,
        metavar="N",
        help="Cap video length to N seconds (use all frames if omitted)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output video path (default: <dataset_stem>_visualization.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Output video FPS (default: 60)",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: dataset file not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    data = np.load(args.dataset, allow_pickle=False)
    if "observations" not in data:
        print("Error: dataset must contain 'observations' array.", file=sys.stderr)
        sys.exit(1)
    if "rewards" not in data:
        print("Error: dataset must contain 'rewards' array.", file=sys.stderr)
        sys.exit(1)
    if "terminals" not in data:
        print("Error: dataset must contain 'terminals' array.", file=sys.stderr)
        sys.exit(1)

    observations = data["observations"]
    rewards = data["rewards"]
    terminals = data["terminals"]
    returns = data["returns"] if "returns" in data else None
    T = len(observations)
    if T == 0:
        print("Error: dataset has no frames.", file=sys.stderr)
        sys.exit(1)
    if len(rewards) != T or len(terminals) != T:
        print(
            f"Error: observations length {T}, rewards {len(rewards)}, terminals {len(terminals)}; lengths must match.",
            file=sys.stderr,
        )
        sys.exit(1)
    if returns is not None and len(returns) != T:
        print(
            f"Error: returns length {len(returns)} does not match observations length {T}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.cap_length is not None:
        max_frames = int(args.cap_length * args.fps)
        N = min(T, max_frames)
    else:
        N = T

    out_path = args.output
    if out_path is None:
        out_path = args.dataset.parent / (args.dataset.stem + "_visualization.mp4")

    writer = imageio.get_writer(str(out_path), fps=args.fps, codec="libx264")
    try:
        for i in range(N):
            if terminals[i]:
                print(f"Terminal at frame {i}")
            return_val = float(returns[i]) if returns is not None else None
            frame = draw_overlay(
                observations[i].copy(),
                float(rewards[i]),
                bool(terminals[i]),
                return_val=return_val,
            )
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"Wrote {N} frames to {out_path}")


if __name__ == "__main__":
    main()
