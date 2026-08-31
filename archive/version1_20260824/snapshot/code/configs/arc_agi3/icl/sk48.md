# SK48 Few-Shot Examples

These examples show successful action decisions in SK48. The agent must thread items
through a claw-arm mechanism by moving a pink-black frame and extending/retracting a rod.

---

## Example 1 — Align frame vertically then extend rod

**Situation:** 64×64 grid. Pink-black frame (col 28-35, row 18-26) is 3 rows above a
cyan bead at (col 31, row 29). Rod is fully retracted to the right.

**Action chosen:** `ACTION2` (move frame down)

**Reasoning:** The frame needs to descend to align row 26 with the bead at row 29.
Moving down 3 times will center the rod slot on the bead before extending.

---

## Example 2 — Extend rod to capture bead

**Situation:** Pink-black frame aligned — bead at (col 31, row 26) is exactly in the
rod slot (row 26). Rod tip is at col 40, bead is at col 31, so bead is to the left.
Available: ACTION1, ACTION2, ACTION3, ACTION4.

**Action chosen:** `ACTION3` (extend/retract rod leftward)

**Reasoning:** The bead is in the correct row. Extending left will push the rod tip
into the bead and thread it. Use ACTION3 until the bead moves with the rod.

---

## Example 3 — Avoid redundant moves

**Situation:** Frame has been moved up 5 times but the grid has not changed (bead still
at row 40). No-effect count for ACTION1 = 5.

**Action chosen:** `ACTION2` (move frame down)

**Reasoning:** Moving up repeatedly with no effect means the frame is already at its
upper limit or the bead is below. Reverse direction and try moving down toward the bead.
