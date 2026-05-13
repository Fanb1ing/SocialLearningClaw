from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


def objects_to_concepts(objects: List[Dict], level: int, step: int) -> List[Concept]:
    """Convert extracted grid objects to Schema Concepts."""
    concepts: List[Concept] = []
    for i, obj in enumerate(objects):
        cname = color_name(obj["color"])
        cid = f"obj_l{level}_s{step}_{i}"
        desc = (
            f"{cname} object at ({obj['top_left'][0]},{obj['top_left'][1]}) to "
            f"({obj['bottom_right'][0]},{obj['bottom_right'][1]}), "
            f"area={obj['area']}, centroid=({obj['centroid'][0]:.1f},{obj['centroid'][1]:.1f})"
        )
        concepts.append(
            Concept(
                id=cid,
                name=f"{cname}Blob_{i}",
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


def diff_objects_to_rules(
    prev_objects: List[Dict],
    prev_concepts: List[Concept],
    curr_objects: List[Dict],
    curr_concepts: List[Concept],
    action_name: str,
) -> List[Relation]:
    """Compare two frames and infer transformation rules.

    Heuristic: if an object disappeared / appeared / moved / changed color,
    create a rule relation like 'ACTION_X + RedBlob_0 -> disappears'.
    """
    # Simplified: just record that action was applied in this context.
    # Full rule mining would need object tracking across frames.
    rules: List[Relation] = []
    for pc in prev_concepts:
        for cc in curr_concepts:
            # If same approximate position -> possibly same object transformed
            rules.append(
                Relation(
                    source=pc.id,
                    target=cc.id,
                    relation_type=f"transformed_by_{action_name}",
                    weight=0.5,
                    evidence=[{"action": action_name}],
                )
            )
    return rules
