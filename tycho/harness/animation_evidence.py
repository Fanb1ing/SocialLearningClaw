"""Animation evidence filtering for ARC-AGI-3 frame sequences.

The ARC engine may return multiple numeric grids for a single action. Most are
animation frames that add little model-relevant evidence. This module implements
a small deterministic filter for the opposite case: transition frames containing
evidence not easily recoverable from the next playable grid alone.

The model is deliberately modest and game-agnostic:

* exact consecutive duplicate grids are dropped losslessly;
* simple local motion is suppressed;
* non-local/eventful sequences get a small farthest-first keyframe summary.

It does not interpret game semantics and does not touch the executable world
model. It just decides whether a transient contact sheet would be worth showing
to a summarizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AnimationEvidenceConfig:
    """Fixed-budget, zero-shot settings for animation evidence extraction.

    The constants are normalized by grid area, so they are not tied to 64x64
    boards. They are intentionally conservative: only local, low-residual motion
    is suppressed; broad changes, border novelty, and intermediate states far
    from both endpoints are surfaced.
    """

    max_keyframes: int = 6
    local_bbox_frac: float = 0.18
    local_max_step_frac: float = 0.04
    event_bbox_frac: float = 0.30
    event_max_step_frac: float = 0.08
    event_mid_endpoint_frac: float = 0.12
    event_border_frac: float = 0.04
    retracting_path_endpoint_ratio: float = 3.5
    retracting_max_step_frac: float = 0.05
    retracting_bbox_frac: float = 0.12


@dataclass
class AnimationDecision:
    surface: bool
    category: str
    reasons: list[str]
    original_frame_count: int
    unique_frame_count: int
    selected_original_indices: list[int]
    features: dict[str, float | int | str]

    def to_json(self) -> dict:
        return asdict(self)


def _as_grid_array(grid) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int16)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D grid, got shape {arr.shape}")
    return arr


def as_grid_sequence(frames: Iterable) -> list[np.ndarray]:
    return [_as_grid_array(g) for g in frames]


def dedupe_consecutive(frames: Sequence[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Drop exact consecutive duplicates, returning grids and original indices."""

    out: list[np.ndarray] = []
    indices: list[int] = []
    for i, grid in enumerate(frames):
        if not out or not np.array_equal(grid, out[-1]):
            out.append(grid)
            indices.append(i)
    return out, indices


def _diff_cells(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _bbox_area(mask: np.ndarray) -> int:
    if not bool(mask.any()):
        return 0
    rows, cols = np.nonzero(mask)
    return int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))


def _border_changed(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return int(np.count_nonzero(mask & border))


def _border_counts(mask: np.ndarray) -> dict[str, int]:
    if mask.size == 0:
        return {"top": 0, "bottom": 0, "left": 0, "right": 0}
    return {
        "top": int(np.count_nonzero(mask[0, :])),
        "bottom": int(np.count_nonzero(mask[-1, :])),
        "left": int(np.count_nonzero(mask[:, 0])),
        "right": int(np.count_nonzero(mask[:, -1])),
    }


def _dominant_border(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    side, count = max(counts.items(), key=lambda kv: kv[1])
    return side if count > 0 else "none"


def _border_cell_count(shape: tuple[int, int]) -> int:
    h, w = shape
    if h <= 0 or w <= 0:
        return 1
    if h == 1:
        return w
    if w == 1:
        return h
    return 2 * h + 2 * w - 4


def _translation_hint(first: np.ndarray, last: np.ndarray, max_shift: int = 8) -> tuple[str, float]:
    """Cheap diagnostic: best small translation overlap between endpoints.

    This is only a label/hint, not a trigger. Constant backgrounds make
    translation scores noisy, so the classifier relies on residual features.
    """

    best = (0, 0, -1.0)
    h, w = first.shape
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            r0a = max(0, -dr)
            r1a = min(h, h - dr)
            c0a = max(0, -dc)
            c1a = min(w, w - dc)
            if r1a <= r0a or c1a <= c0a:
                continue
            a = first[r0a:r1a, c0a:c1a]
            b = last[r0a + dr:r1a + dr, c0a + dc:c1a + dc]
            score = float(np.mean(a == b))
            if score > best[2]:
                best = (dr, dc, score)
    dr, dc, score = best
    return f"dr={dr},dc={dc}", score


def _features(frames: Sequence[np.ndarray]) -> dict[str, float | int | str]:
    first, last = frames[0], frames[-1]
    area = max(1, int(first.size))
    union = np.zeros(first.shape, dtype=bool)
    step_diffs: list[int] = []
    for a, b in zip(frames, frames[1:]):
        changed = a != b
        union |= changed
        step_diffs.append(int(np.count_nonzero(changed)))

    endpoint = _diff_cells(first, last)
    max_step = max(step_diffs, default=0)
    path = sum(step_diffs)
    mid_min_endpoint = 0
    for grid in frames[1:-1]:
        mid_min_endpoint = max(
            mid_min_endpoint,
            min(_diff_cells(grid, first), _diff_cells(grid, last)),
        )
    shift, shift_score = _translation_hint(first, last)
    border_counts = _border_counts(union)
    return {
        "grid_area": area,
        "endpoint_cells": endpoint,
        "endpoint_frac": endpoint / area,
        "max_step_cells": max_step,
        "max_step_frac": max_step / area,
        "path_cells": path,
        "path_endpoint_ratio": path / max(1, endpoint),
        "union_cells": int(np.count_nonzero(union)),
        "union_bbox_cells": _bbox_area(union),
        "union_bbox_frac": _bbox_area(union) / area,
        "border_cells": _border_changed(union),
        "border_frac": _border_changed(union) / _border_cell_count(first.shape),
        "border_top_cells": border_counts["top"],
        "border_bottom_cells": border_counts["bottom"],
        "border_left_cells": border_counts["left"],
        "border_right_cells": border_counts["right"],
        "dominant_border": _dominant_border(border_counts),
        "mid_min_endpoint_cells": mid_min_endpoint,
        "mid_min_endpoint_frac": mid_min_endpoint / area,
        "best_endpoint_translation": shift,
        "best_endpoint_translation_score": shift_score,
    }


def _count_bucket(n: int) -> str:
    if n <= 5:
        return "3-5"
    if n <= 12:
        return "6-12"
    if n <= 32:
        return "13-32"
    return "33+"


def _frac_bucket(x: float) -> str:
    if x < 0.02:
        return "tiny"
    if x < 0.08:
        return "small"
    if x < 0.30:
        return "medium"
    if x < 0.75:
        return "large"
    return "full"


def animation_signature(decision: AnimationDecision, *, action: str | None = None,
                        terminal: str = "nonterminal") -> tuple[tuple[str, str], ...]:
    """Coarse exact-match key for reusing an animation summary.

    This is deliberately not a pixel hash. It groups transitions by the kind of
    evidence they contain: category, rough frame count, spatial footprint, and
    border involvement. The action name is included without click coordinates.
    """

    feats = decision.features
    return tuple(sorted({
        "terminal": terminal,
        "category": decision.category,
        "action": action or "",
        "frames": _count_bucket(decision.original_frame_count),
        "bbox": _frac_bucket(float(feats.get("union_bbox_frac", 0.0))),
        "max_step": _frac_bucket(float(feats.get("max_step_frac", 0.0))),
        "mid": _frac_bucket(float(feats.get("mid_min_endpoint_frac", 0.0))),
        "border": str(feats.get("dominant_border", "none")),
    }.items()))


def select_keyframes(frames: Sequence[np.ndarray], original_indices: Sequence[int],
                     max_keyframes: int) -> list[int]:
    """Return original frame indices for a small diverse keyframe set.

    Farthest-first is a deterministic approximation to diversity/coverage:
    start with endpoints, then add the frame that is farthest from the already
    selected set under Hamming grid distance.
    """

    n = len(frames)
    if n <= max_keyframes:
        return list(original_indices)
    selected = [0, n - 1]
    while len(selected) < max_keyframes:
        best_i = None
        best_dist = -1
        for i, grid in enumerate(frames):
            if i in selected:
                continue
            dist = min(_diff_cells(grid, frames[j]) for j in selected)
            if dist > best_dist:
                best_dist = dist
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
    return [original_indices[i] for i in sorted(selected)]


def analyze_frame_sequence(frames_like: Iterable, config: AnimationEvidenceConfig | None = None) -> AnimationDecision:
    """Classify a returned engine frame list for optional summarization."""

    config = config or AnimationEvidenceConfig()
    original = as_grid_sequence(frames_like)
    if not original:
        return AnimationDecision(
            surface=False,
            category="empty",
            reasons=["no frames"],
            original_frame_count=0,
            unique_frame_count=0,
            selected_original_indices=[],
            features={},
        )
    frames, original_indices = dedupe_consecutive(original)
    if len(frames) <= 2:
        return AnimationDecision(
            surface=False,
            category="endpoint_sufficient",
            reasons=["sequence has no distinct interior numeric frame after exact dedupe"],
            original_frame_count=len(original),
            unique_frame_count=len(frames),
            selected_original_indices=list(original_indices),
            features=_features(frames) if len(frames) >= 2 else {"grid_area": int(frames[0].size)},
        )

    feats = _features(frames)
    reasons: list[str] = []
    is_local_motion = (
        float(feats["union_bbox_frac"]) <= config.local_bbox_frac
        and float(feats["max_step_frac"]) <= config.local_max_step_frac
    )
    if is_local_motion:
        return AnimationDecision(
            surface=False,
            category="local_motion",
            reasons=[
                "changed cells stay within a small bounding box",
                "largest consecutive numeric-grid delta is small",
            ],
            original_frame_count=len(original),
            unique_frame_count=len(frames),
            selected_original_indices=select_keyframes(frames, original_indices, config.max_keyframes),
            features=feats,
        )

    if float(feats["max_step_frac"]) >= config.event_max_step_frac:
        reasons.append("large consecutive numeric-grid delta")
    if float(feats["union_bbox_frac"]) >= config.event_bbox_frac:
        reasons.append("changes occupy a broad grid region")
    if float(feats["mid_min_endpoint_frac"]) >= config.event_mid_endpoint_frac:
        reasons.append("interior frames are far from both endpoints")
    if float(feats["border_frac"]) >= config.event_border_frac:
        reasons.append("border changes suggest viewport/hidden-world reveal")

    if reasons:
        category = "eventful_transition"
        if "border changes suggest viewport/hidden-world reveal" in reasons:
            category = "viewport_or_reveal"
        elif "interior frames are far from both endpoints" in reasons:
            category = "nonredundant_intermediate"
        return AnimationDecision(
            surface=True,
            category=category,
            reasons=reasons,
            original_frame_count=len(original),
            unique_frame_count=len(frames),
            selected_original_indices=select_keyframes(frames, original_indices, config.max_keyframes),
            features=feats,
        )

    is_retracting_transient = (
        float(feats["path_endpoint_ratio"]) >= config.retracting_path_endpoint_ratio
        and float(feats["max_step_frac"]) >= config.retracting_max_step_frac
        and float(feats["union_bbox_frac"]) >= config.retracting_bbox_frac
    )
    if is_retracting_transient:
        return AnimationDecision(
            surface=True,
            category="retracting_transient",
            reasons=[
                "intermediate frames changed much more than the endpoint delta",
                "changed cells covered a non-local region",
                "largest animation step suggests a transient event",
            ],
            original_frame_count=len(original),
            unique_frame_count=len(frames),
            selected_original_indices=select_keyframes(frames, original_indices, config.max_keyframes),
            features=feats,
        )

    return AnimationDecision(
        surface=False,
        category="low_residual_animation",
        reasons=["not local enough to label as simple motion, but below event-evidence gates"],
        original_frame_count=len(original),
        unique_frame_count=len(frames),
        selected_original_indices=select_keyframes(frames, original_indices, config.max_keyframes),
        features=feats,
    )


def _contact_sheet_image(frames_like: Sequence, selected_indices: Sequence[int],
                         *, scale: int = 8, title: str = ""):
    from PIL import Image, ImageDraw
    from arc_agi.rendering import frame_to_rgb_array

    frames = as_grid_sequence(frames_like)
    selected = [i for i in selected_indices if 0 <= i < len(frames)]
    if not selected:
        selected = [0]
    tiles: list[Image.Image] = []
    label_h = 18
    for idx in selected:
        rgb = np.asarray(frame_to_rgb_array(0, frames[idx], scale=scale), dtype=np.uint8)
        img = Image.fromarray(rgb)
        tile = Image.new("RGB", (img.width, img.height + label_h), "white")
        tile.paste(img, (0, label_h))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 2), f"frame {idx}", fill=(0, 0, 0))
        tiles.append(tile)

    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    pad = 8
    title_h = 22 if title else 0
    w = cols * tiles[0].width + (cols - 1) * pad
    h = rows * tiles[0].height + (rows - 1) * pad + title_h
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((4, 3), title[:160], fill=(0, 0, 0))
    for n, tile in enumerate(tiles):
        r, c = divmod(n, cols)
        x = c * (tiles[0].width + pad)
        y = title_h + r * (tiles[0].height + pad)
        sheet.paste(tile, (x, y))
    return sheet


def render_contact_sheet(frames_like: Sequence, selected_indices: Sequence[int], out_path: Path,
                         *, scale: int = 8, title: str = "") -> None:
    """Render selected numeric grids as one PNG contact sheet."""

    sheet = _contact_sheet_image(frames_like, selected_indices, scale=scale, title=title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def contact_sheet_png_bytes(frames_like: Sequence, selected_indices: Sequence[int],
                            *, scale: int = 8, title: str = "") -> bytes:
    sheet = _contact_sheet_image(frames_like, selected_indices, scale=scale, title=title)
    buf = BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


def selected_frame_jpeg_bytes(frames_like: Sequence, selected_indices: Sequence[int],
                              *, scale: int = 32, quality: int = 95) -> list[bytes]:
    """Render selected numeric grids as high-quality JPEG frames for vLLM video input.

    vLLM's OpenAI-compatible pre-extracted video path expects JPEG frame payloads.
    Keep the scale caller-controlled so Qwen can use its normal 32 px/cell path.
    """

    from PIL import Image
    from arc_agi.rendering import frame_to_rgb_array

    frames = as_grid_sequence(frames_like)
    selected = [i for i in selected_indices if 0 <= i < len(frames)]
    if not selected:
        selected = [0]
    out: list[bytes] = []
    for idx in selected:
        rgb = np.asarray(frame_to_rgb_array(0, frames[idx], scale=scale), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, subsampling=0)
        out.append(buf.getvalue())
    return out


def selected_frame_png_bytes(frames_like: Sequence, selected_indices: Sequence[int],
                             *, scale: int = 32) -> list[bytes]:
    """Render selected numeric grids as separate PNG frame images."""

    from PIL import Image
    from arc_agi.rendering import frame_to_rgb_array

    frames = as_grid_sequence(frames_like)
    selected = [i for i in selected_indices if 0 <= i < len(frames)]
    if not selected:
        selected = [0]
    out: list[bytes] = []
    for idx in selected:
        rgb = np.asarray(frame_to_rgb_array(0, frames[idx], scale=scale), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def selected_frame_gif_bytes(frames_like: Sequence, selected_indices: Sequence[int],
                             *, scale: int = 32, duration_ms: int = 500) -> bytes:
    """Render selected numeric grids as an animated GIF for video-capable servers.

    This is a dependency-light fallback for servers such as vllm-mlx that accept
    data:video/* payloads but do not accept vLLM's pre-extracted JPEG-frame URI.
    ARC grids use a small fixed palette, so GIF preserves the cell colors well.
    """

    from PIL import Image
    from arc_agi.rendering import frame_to_rgb_array

    frames = as_grid_sequence(frames_like)
    selected = [i for i in selected_indices if 0 <= i < len(frames)]
    if not selected:
        selected = [0]
    images: list[Image.Image] = []
    for idx in selected:
        rgb = np.asarray(frame_to_rgb_array(0, frames[idx], scale=scale), dtype=np.uint8)
        images.append(Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE, colors=32))
    buf = BytesIO()
    images[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()
