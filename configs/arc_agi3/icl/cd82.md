# CD82 Few-Shot Examples

These examples show correct action decisions in CD82, the paint-dyeing puzzle.
The goal is to paint the central target object to match the reference pattern in the top-left corner.

---

## Example 1 — Select the correct dye color before painting

**Situation:** 64×64 grid. Reference (top-left, yellow border) shows: top half = BLACK, bottom half = WHITE.
Target object is currently fully black (unpainted). Dyeing container is positioned directly **above** the target (open side facing down). Container interior is **white** — current dye is white.
The black swatch is visible at approximately (col 40, row 3) in the green top strip.

**Action chosen:** `ACTION6` at `{"x": 40, "y": 3}`

**Reasoning:** The container is already above the target, which is the correct position to paint the top half. However, the current dye is WHITE and the top half needs BLACK. I must switch to the black dye first by clicking the black swatch before applying ACTION5, otherwise I will paint the wrong color.

---

## Example 2 — Paint the top region with correct dye

**Situation:** Reference shows: top half = BLACK, bottom half = WHITE. Container is positioned directly **above** the target, open side facing down. Current dye is now **black** (just switched via ACTION6). Target top half is still unpainted (black background color).

**Action chosen:** `ACTION5`

**Reasoning:** The container is in the correct orbital position (directly above, open side pointing down toward the target's top half) and the correct dye color (black) is selected. ACTION5 will extend the container downward, overlapping and painting the top half of the target black — matching the reference.

---

## Example 3 — Orbit container to the opposite side before painting

**Situation:** Reference shows: top half = BLACK, bottom half = WHITE. Top half is now correctly painted black. Container is still positioned **above** the target. Current dye is black. Need to paint the bottom half white.

**Action chosen:** `ACTION4`, then `ACTION2`, `ACTION2`, `ACTION3` (4 steps to reach the bottom)

**Reasoning:** The container must orbit from 12 o'clock (above) to 6 o'clock (below), where its open side faces up toward the bottom half of the target. The clockwise path from 12 o'clock takes 4 steps: ACTION4 (to upper-right) → ACTION2 (to right) → ACTION2 (to lower-right) → ACTION3 (to below). Once at 6 o'clock, I switch to white dye (ACTION6 at the white swatch) and press ACTION5 to paint the bottom half white.
