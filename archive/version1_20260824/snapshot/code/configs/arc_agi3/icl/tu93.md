# TU93 Few-Shot Examples

These examples show correct action decisions in TU93.

---

## Example 1 — Follow the path, avoid gray cells

**Situation:** 64×64 grid. Player (maroon 3×3 block) is at approximately (col 15–17, row 15–17).
Yellow notch pixel is at (col 17, row 16) — player faces RIGHT.
Red-black path continues to the RIGHT from the player's current position (path cells visible at col 18–24, row 16).
Magenta goal is far to the bottom-right at approximately (col 45–47, row 45–47).

**Action chosen:** `ACTION4` (move right)

**Reasoning:** The yellow notch faces right and the red-black path continues to the right. Moving right follows the path toward the goal. The gray cells above and below are walls; only the rightward path cells are walkable at this position.

---

## Example 2 — Avoid enemy's facing direction, approach from the side

**Situation:** Player (maroon) is at (col 30–32, row 20–22). The path goes right, but at (col 36, row 21)
there is an **enemy** (similar 3×3 block) with its notch facing LEFT (at col 34, row 21).
This means the cell at (col 33, row 21) — directly in front of the enemy — is lethal.

**Action chosen:** `ACTION2` (move down)

**Reasoning:** Moving right would put the player at (col 33, row 21) which is directly in front of the enemy's notch — instant GAME_OVER. Instead, moving down follows an alternate path segment that approaches the enemy from the side or rear, which is safe. After reaching the cell beside or behind the enemy, I can step onto it to eliminate it and continue toward the goal.

---

## Example 3 — Backtrack when stuck at a dead end

**Situation:** Player is at (col 22–24, row 38–40). The path in front has no more red-black cells in the forward direction (down). No path to the left. To the right is gray (wall).

**Action chosen:** `ACTION1` (move up)

**Reasoning:** The player has reached a dead end in this direction. The only walkable path cells are back the way I came (upward). I must backtrack to find the correct branching point in the maze that leads toward the magenta goal at the bottom-right. Continuing to press ACTION2 (down) would have no effect since there are no path cells below.
