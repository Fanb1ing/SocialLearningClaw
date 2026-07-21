from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .graph import Concept, Relation, SchemaGraph


def _simple_label(grid: np.ndarray, value: int) -> Tuple[np.ndarray, int]:
    """Simple connected-component labeling for a single value using BFS."""
    h, w = grid.shape
    mask = grid == value
    visited = np.zeros_like(mask, dtype=bool)
    labels = np.zeros_like(mask, dtype=int)
    label_id = 0

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            label_id += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                labels[cy, cx] = label_id
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return labels, label_id


def extract_grid_objects(grid: np.ndarray) -> List[Dict]:
    """Extract connected-component objects from grid without scipy."""
    objects: List[Dict] = []
    unique_vals = sorted(set(int(v) for v in grid.flatten()))

    for val in unique_vals:
        if val == 0:
            continue
        labels, num_features = _simple_label(grid, val)
        for i in range(1, num_features + 1):
            ys, xs = np.where(labels == i)
            if len(xs) == 0:
                continue
            obj = {
                "color": val,
                "top_left": (int(xs.min()), int(ys.min())),
                "bottom_right": (int(xs.max()), int(ys.max())),
                "area": int(len(xs)),
                "centroid": (float(xs.mean()), float(ys.mean())),
            }
            objects.append(obj)
    return objects


def color_name(color: int) -> str:
    """Map ARC color index to a descriptive name."""
    palette = {
        0: "Black",
        1: "Blue",
        2: "Red",
        3: "Green",
        4: "Yellow",
        5: "Gray",
        6: "Pink",
        7: "Orange",
        8: "Cyan",
        9: "Maroon",
        10: "Beige",
        11: "Lime",
        12: "Indigo",
        13: "Brown",
        14: "Magenta",
        15: "White",
    }
    return palette.get(color, f"Color{color}")


def objects_to_concepts(objects: List[Dict], level: int, step: int = 0) -> List[Concept]:
    """Convert extracted grid objects to Schema Concepts.

    IDs are stable across steps: obj_l{level}_{colorname}_{per_color_index}.
    Repeated calls for the same level update concept descriptions in-place
    (via SchemaGraph.add_concept overwrite) instead of accumulating new entries.
    The ``step`` parameter is kept for backward compatibility but no longer
    affects the generated IDs.
    """
    concepts: List[Concept] = []
    color_counter: Dict[int, int] = {}
    for obj in objects:
        c = obj["color"]
        idx = color_counter.get(c, 0)
        color_counter[c] = idx + 1
        cname = color_name(c)
        cid = f"obj_l{level}_{cname.lower()}_{idx}"
        desc = (
            f"{cname} object at ({obj['top_left'][0]},{obj['top_left'][1]}) to "
            f"({obj['bottom_right'][0]},{obj['bottom_right'][1]}), "
            f"area={obj['area']}, centroid=({obj['centroid'][0]:.1f},{obj['centroid'][1]:.1f})"
        )
        concepts.append(
            Concept(
                id=cid,
                name=f"{cname}Blob_{idx}",
                description=desc,
                category=f"level_{level}",
                confidence=0.6,
                source="agent_observation",
                created_at="",
            )
        )
    return concepts


def build_spatial_relations(objects: List[Dict], concepts: List[Concept], max_neighbors: int = 3) -> List[Relation]:
    """Build spatial relations between objects based on centroids.

    Each object only connects to its max_neighbors nearest objects to keep
    the schema graph sparse and prompts manageable.
    """
    import math

    n = len(objects)
    if n <= 1:
        return []

    # Compute pairwise distances
    dists = []
    for i in range(n):
        xi, yi = objects[i]["centroid"]
        for j in range(i + 1, n):
            xj, yj = objects[j]["centroid"]
            d = math.hypot(xj - xi, yj - yi)
            dists.append((d, i, j))

    # For each object, keep only nearest max_neighbors
    neighbors: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(n)}
    for d, i, j in dists:
        neighbors[i].append((d, j))
        neighbors[j].append((d, i))

    for i in range(n):
        neighbors[i].sort(key=lambda x: x[0])
        neighbors[i] = neighbors[i][:max_neighbors]

    # Build relations from kept neighbors
    relations: List[Relation] = []
    added: set = set()
    for i in range(n):
        for _, j in neighbors[i]:
            key = (min(i, j), max(i, j))
            if key in added:
                continue
            added.add(key)
            xi, yi = objects[i]["centroid"]
            xj, yj = objects[j]["centroid"]
            dx = xj - xi
            dy = yj - yi
            if abs(dx) > abs(dy):
                rel_type = "right_of" if dx > 0 else "left_of"
            else:
                rel_type = "below" if dy > 0 else "above"
            relations.append(
                Relation(
                    source=concepts[i].id,
                    target=concepts[j].id,
                    relation_type=rel_type,
                    weight=0.5,
                    evidence=[],
                )
            )
    return relations


def compute_grid_diff(pre_grid: np.ndarray, post_grid: np.ndarray) -> Tuple[bool, List[Dict]]:
    """Compare pre- and post-action grids pixel-wise.

    Returns (changed, regions) where each region dict contains:
    {"top_left": (x, y), "bottom_right": (x, y), "color_before": int, "color_after": int}
    """
    if pre_grid is None or post_grid is None:
        return False, []
    if pre_grid.shape != post_grid.shape:
        return True, [
            {
                "top_left": (0, 0),
                "bottom_right": (
                    max(pre_grid.shape[1], post_grid.shape[1]) - 1,
                    max(pre_grid.shape[0], post_grid.shape[0]) - 1,
                ),
                "color_before": -1,
                "color_after": -1,
                "shape_before": list(pre_grid.shape),
                "shape_after": list(post_grid.shape),
            }
        ]

    h, w = pre_grid.shape
    changed_pixels = []
    for y in range(h):
        for x in range(w):
            if pre_grid[y, x] != post_grid[y, x]:
                changed_pixels.append({
                    "x": x,
                    "y": y,
                    "color_before": int(pre_grid[y, x]),
                    "color_after": int(post_grid[y, x]),
                })

    if not changed_pixels:
        return False, []

    # Build coarse regions from changed pixels (group by color_before/color_after pairs)
    regions: List[Dict] = []
    # Simple approach: one bounding box for all changed pixels + one per color transition
    xs = [p["x"] for p in changed_pixels]
    ys = [p["y"] for p in changed_pixels]
    regions.append({
        "top_left": (min(xs), min(ys)),
        "bottom_right": (max(xs), max(ys)),
        "color_before": -1,
        "color_after": -1,
    })
    return True, regions


def _summarize_state(objects: List[Dict], concepts: List[Concept], max_n: int = 3) -> str:
    """Compact before/after state summary for action concept descriptions."""
    parts = []
    for i, (obj, c) in enumerate(zip(objects, concepts)):
        if i >= max_n:
            parts.append(f"+{len(objects) - max_n} more")
            break
        tl = obj.get("top_left", ("?", "?"))
        br = obj.get("bottom_right", ("?", "?"))
        area = obj.get("area", "?")
        parts.append(f"{c.name}[col={tl[0]}-{br[0]},row={tl[1]}-{br[1]},area={area}]")
    return "; ".join(parts) if parts else "none"


def build_action_effect_concepts_and_relations(
    action_name: str,
    action_data: Dict[str, Any],
    step: int,
    level: int,
    grid_changed: bool,
    changed_regions: List[Dict],
    pre_objects: List[Dict],
    pre_concepts: List[Concept],
    post_objects: List[Dict],
    post_concepts: List[Concept],
) -> Tuple[List[Concept], List[Relation]]:
    """Create action concept + effect relations for schema.

    If grid_changed is False: emits a no_effect relation.
    If grid_changed is True: emits relations linking the action to affected objects.
    """
    import math

    concepts: List[Concept] = []
    relations: List[Relation] = []

    # Build rich action description with before/after state
    coord_str = ""
    if action_data:
        coord_str = f" at (col={action_data.get('x', '?')},row={action_data.get('y', '?')})"

    pre_str = _summarize_state(pre_objects, pre_concepts) if pre_objects else "unknown(no prior observation)"
    post_str = _summarize_state(post_objects, post_concepts) if post_objects else "unknown"

    changed_region_str = ""
    if grid_changed and changed_regions:
        r = changed_regions[0]
        tl, br = r["top_left"], r["bottom_right"]
        changed_region_str = f" Affected region: col={tl[0]}-{br[0]},row={tl[1]}-{br[1]}."

    description = (
        f"{action_name}{coord_str}: grid_changed={grid_changed}.{changed_region_str} "
        f"Before: {pre_str}. After: {post_str}"
    )

    level_category = f"level_{level}"
    action_cid = f"action_l{level}_s{step}_{action_name}"
    action_concept = Concept(
        id=action_cid,
        name=f"Action_{action_name}{coord_str.replace(' ', '_').replace('=', '')}",
        description=description,
        category=level_category,
        confidence=0.7,
        source="action_effect",
        created_at="",
    )
    concepts.append(action_concept)

    if not grid_changed:
        no_effect_cid = f"no_effect_l{level}_s{step}"
        no_effect_concept = Concept(
            id=no_effect_cid,
            name="NoEffect",
            description=f"Action {action_name}{coord_str} caused no visible grid change",
            category=level_category,
            confidence=0.9,
            source="action_effect",
            created_at="",
        )
        concepts.append(no_effect_concept)
        relations.append(
            Relation(
                source=action_cid,
                target=no_effect_cid,
                relation_type="no_effect",
                weight=0.9,
                evidence=[{"action": action_name, "step": step}],
            )
        )
        return concepts, relations

    # Grid changed: try to map changed regions to post-action objects
    affected_obj_ids: set = set()
    for region in changed_regions:
        r_x1, r_y1 = region["top_left"]
        r_x2, r_y2 = region["bottom_right"]
        for j, obj in enumerate(post_objects):
            o_x1, o_y1 = obj["top_left"]
            o_x2, o_y2 = obj["bottom_right"]
            # Check overlap
            if not (r_x2 < o_x1 or r_x1 > o_x2 or r_y2 < o_y1 or r_y1 > o_y2):
                affected_obj_ids.add(j)

    for j in affected_obj_ids:
        if j < len(post_concepts):
            post_c = post_concepts[j]
            relations.append(
                Relation(
                    source=action_cid,
                    target=post_c.id,
                    relation_type="affected",
                    weight=0.7,
                    evidence=[{"action": action_name, "step": step}],
                )
            )

    return concepts, relations


_DEFAULT_COLOR_PALETTE = {
    0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
    5: "Gray", 6: "Pink", 7: "Orange", 8: "Cyan", 9: "Maroon",
    10: "Beige", 11: "Lime", 12: "Indigo", 13: "Brown", 14: "Magenta", 15: "White",
}


_CONCEPT_EXTRACTION_PROMPT_TEMPLATE = (
    "You are analyzing an ARC-AGI-3 grid image.\n"
    "The grid is {h} rows x {w} columns. Each cell is a single unit in grid coordinates.\n"
    "Coordinate system:\n"
    "  - col (x): 0 = leftmost column, increases to the RIGHT, max = {w_max}\n"
    "  - row (y): 0 = topmost row, increases DOWNWARD, max = {h_max}\n"
    "  - All positions use [col, row] = [x, y] order.\n"
    "  - The image has sparse gridlines every 8 cells to help locate positions.\n"
    "IMPORTANT: Report ALL coordinates as grid-cell indices, NOT pixel coordinates.\n\n"
    "Identify the distinct objects or patterns in the grid. "
    "Ignore single isolated pixels (noise). "
    "Group adjacent same-color pixels into meaningful objects. "
    "If small same-color pixels form a line or shape, describe them as ONE object.\n\n"
    "Output ONLY a JSON array with at most 10 most important objects:\n"
    '[\n'
    '  {"name": "Agent", "color": "Pink", "top_left": [2, 3], "bottom_right": [4, 5],\n'
    '   "description": "Pink blob with black center — top_left=[col=2,row=3], bottom_right=[col=4,row=5]"},\n'
    '  {"name": "RedStick", "color": "Red", "top_left": [5, 5], "bottom_right": [5, 8],\n'
    '   "description": "Vertical red line at col=5, rows 5-8"}\n'
    "]\n\n"
    "Be concise. Return valid JSON only."
)


def llm_extract_grid_concepts(
    grid_img, agent, level: int, step: int, grid_shape: Optional[Tuple[int, int]] = None
) -> Tuple[List[Concept], List[Relation]]:
    """Use a vision LLM to extract meaningful concepts from a grid image.

    Returns (concepts, relations) where relations are spatial relations
    between the extracted concepts.
    """
    import base64
    import io
    import json
    import math
    import warnings

    # Infer grid shape from image if not provided (assume square cell_size=8)
    if grid_shape is None:
        w_px, h_px = grid_img.size
        grid_shape = (h_px // 8, w_px // 8)
    h, w = grid_shape
    h_max, w_max = h - 1, w - 1

    prompt = _CONCEPT_EXTRACTION_PROMPT_TEMPLATE.format(h=h, w=w, h_max=h_max, w_max=w_max)

    # Encode image to base64 data URL
    buffer = io.BytesIO()
    grid_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    image_data_url = f"data:image/png;base64,{img_base64}"

    # Call LLM with image
    attempt = agent.answer(
        prompt=prompt,
        meta={"image_data_url": image_data_url},
    )

    raw = attempt.answer_text.strip()
    concepts: List[Concept] = []

    # Try to parse JSON array
    try:
        # Handle markdown code fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "objects" in parsed:
            parsed = parsed["objects"]
        if not isinstance(parsed, list):
            parsed = [parsed]
    except Exception:
        # Fallback: return empty if parsing fails
        return concepts, []

    for i, obj in enumerate(parsed):
        if not isinstance(obj, dict):
            continue
        name = obj.get("name", f"Object_{i}")
        color = obj.get("color", "Unknown")
        desc = obj.get("description", "")
        tl = obj.get("top_left", [0, 0])
        br = obj.get("bottom_right", [0, 0])

        # Validate coordinates are within grid bounds
        try:
            tl_x, tl_y = int(tl[0]), int(tl[1])
            br_x, br_y = int(br[0]), int(br[1])
        except Exception:
            warnings.warn(f"[LLM Concept] Skipping object '{name}' due to non-integer coordinates: tl={tl}, br={br}")
            continue

        if not (0 <= tl_x <= w_max and 0 <= tl_y <= h_max and 0 <= br_x <= w_max and 0 <= br_y <= h_max):
            warnings.warn(
                f"[LLM Concept] Skipping object '{name}' with out-of-bounds coordinates "
                f"tl=({tl_x},{tl_y}) br=({br_x},{br_y}) on a {h}x{w} grid"
            )
            continue

        cid = f"llm_l{level}_{name.lower().replace(' ', '_')}_{i}"
        full_desc = (
            f"{color} {name} at ({tl_x},{tl_y}) to ({br_x},{br_y}). {desc}"
        )
        concepts.append(
            Concept(
                id=cid,
                name=f"{color}{name}",
                description=full_desc,
                category=f"level_{level}",
                confidence=0.8,
                source="llm_vision",
                created_at="",
            )
        )

    # Build spatial relations between LLM concepts using their bounding-box centroids
    relations: List[Relation] = []
    n = len(concepts)
    if n <= 1:
        return concepts, relations

    # Build pseudo-objects for relation builder
    pseudo_objects = []
    for c in concepts:
        # Parse top_left/bottom_right from description as fallback
        # We stored them in description; extract centroid from the desc
        # Format: "... at (x,y) to (x,y). ..."
        import re as _re
        m = _re.search(r"at \((\d+),(\d+)\) to \((\d+),(\d+)\)", c.description)
        if m:
            x1, y1, x2, y2 = map(int, m.groups())
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        else:
            centroid = (0.0, 0.0)
        pseudo_objects.append({"centroid": centroid})

    # Nearest-neighbor relations
    dists = []
    for i in range(n):
        xi, yi = pseudo_objects[i]["centroid"]
        for j in range(i + 1, n):
            xj, yj = pseudo_objects[j]["centroid"]
            d = math.hypot(xj - xi, yj - yi)
            dists.append((d, i, j))

    neighbors: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(n)}
    for d, i, j in dists:
        neighbors[i].append((d, j))
        neighbors[j].append((d, i))

    for i in range(n):
        neighbors[i].sort(key=lambda x: x[0])
        neighbors[i] = neighbors[i][:3]  # max 3 neighbors

    added: set = set()
    for i in range(n):
        for _, j in neighbors[i]:
            key = (min(i, j), max(i, j))
            if key in added:
                continue
            added.add(key)
            xi, yi = pseudo_objects[i]["centroid"]
            xj, yj = pseudo_objects[j]["centroid"]
            dx = xj - xi
            dy = yj - yi
            if abs(dx) > abs(dy):
                rel_type = "right_of" if dx > 0 else "left_of"
            else:
                rel_type = "below" if dy > 0 else "above"
            relations.append(
                Relation(
                    source=concepts[i].id,
                    target=concepts[j].id,
                    relation_type=rel_type,
                    weight=0.5,
                    evidence=[],
                )
            )

    return concepts, relations
