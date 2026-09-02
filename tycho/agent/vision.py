"""Per-model visual-encoder facts — the single source of truth for HOW we render the 64x64 frame
for a given model and WHAT we may truthfully tell that model about the image's fidelity.

Two coupled decisions live here so they can never drift apart:
  - render_scale (px per ARC cell): how big to draw the PNG;
  - lossless_cells: whether, at that scale, the model's vision encoder preserves one visual token
    per ARC cell with NO downscale — i.e. whether the image is reliable for EXACT per-cell color.

The prompt's perception sentence is selected from `lossless_cells` (vision.py is imported by the
agent's prompt assembly), so a model that downscales is told "the image is lower-resolution; use the
python tool for exact cells", while a model that renders 1-token-per-cell is told the image is
full-resolution. Previously the resolution claim was hard-wired to whether the text grid was inlined,
which let a downscaling model be told "no information loss" — false. Keyed on the model string only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VisionProfile:
    render_scale: int      # px per ARC cell when rendering the frame PNG
    lossless_cells: bool   # does the encoder keep ~1 visual token / cell at this scale (no downscale)?
    note: str              # short human-readable rationale (logged + stamped in the manifest)


def vision_profile(model: str) -> VisionProfile:
    """The (render_scale, lossless_cells) a model should use. TYCHO_RENDER_SCALE overrides the scale
    (ablations), but lossless_cells then follows the OVERRIDDEN scale where we can reason about it."""
    m = (model or "").lower()
    if "qwen" in m:
        # Qwen3.x VLM: patch_size 16 × spatial_merge 2 = 32 px per merged visual token. Rendering the
        # 64x64 grid at 32 px/cell → 2048x2048 → 128x128 patches → merge → 4096 visual tokens = exactly
        # one per ARC cell if the serving processor is allowed max_pixels >= 4194304 and tokenizer
        # truncation is disabled or raised by the serving configuration.
        # At the default 6 px/cell the same encoder WOULD downscale, so lossless_cells tracks the scale.
        scale = _scale_override(default=32)
        return VisionProfile(scale, lossless_cells=(scale >= 32),
                             note=f"qwen3.x patch16×merge2=32px/token; scale={scale} → "
                                  f"{'1 token/cell, no downscale' if scale >= 32 else 'downscaled'}")
    if "gemma" in m or "gemini" in m:
        # Gemma/Gemini pack the image into a small fixed visual-token budget and downscale hard, so a
        # 64x64 grid loses per-cell fidelity through the encoder regardless of how large we render it.
        return VisionProfile(_scale_override(default=6), lossless_cells=False,
                             note="gemma/gemini: fixed small token budget → image downscaled (lossy per-cell)")
    if "gpt" in m or m.startswith("o"):
        # GPT-5.x vision tiles at 512px; a 64x64 grid is tiled + downscaled, not 1 token/cell. Lossy.
        return VisionProfile(_scale_override(default=6), lossless_cells=False,
                             note="gpt-5.x: 512px tiling → image downscaled (lossy per-cell)")
    # Opus/Claude/unknown: Claude vision resizes to ~1.15 MP; at default 6 px/cell the per-cell color
    # is not reliably recoverable from the image. Treat as lossy → lean on the text grid + python tool.
    return VisionProfile(_scale_override(default=6), lossless_cells=False,
                         note="claude/opus or unknown: image resized → treat as lossy per-cell")


def _scale_override(default: int) -> int:
    """TYCHO_RENDER_SCALE wins when set (ablation knob); else the profile default."""
    v = os.environ.get("TYCHO_RENDER_SCALE", "").strip()
    return int(v) if v else default
