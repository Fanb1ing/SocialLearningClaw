# CD82 Game Rules

## Screen Layout

The screen has three distinct regions:

- **Top-left corner** (yellow border): The **reference target** — a small grid showing the exact color pattern the central object must be painted to match. This is read-only; it does not change.
- **Top strip** (green background): The **color swatch panel**. Contains selectable color blocks (e.g. black square, white square). Click with ACTION6 to switch the dye color.
- **Center play area**: Contains the **dyeing container** (染色缸) and the **target object** directly below it.

---

## Controllable Elements

### 1. Dyeing Container
- A rectangular frame with **red borders on three sides** and one **open side**.
- The **open side always faces the target object** (the center of the play area).
- The **white interior** shows the current dye color.
- Orbits the target object at **8 discrete positions** (every 45°), like a clock hand.
- **Each action moves the container one step in the literal screen direction:**
  - **ACTION1** → move container center **UP** (row decreases)
  - **ACTION2** → move container center **DOWN** (row increases)
  - **ACTION3** → move container center **LEFT** (col decreases)
  - **ACTION4** → move container center **RIGHT** (col increases)
- **ACTION5** → extend the container **toward the target center** once; all cells where the container overlaps the target get painted with the current dye color; then retract.

#### Orbital Position Reference

The container starts at **12 o'clock** (directly above the target, open side facing down).
To advance **clockwise**, use actions in this order — one action per 45° step:

| Current position | Action to go clockwise | → Next position |
|---|---|---|
| 0° — 12 o'clock (above) | **ACTION4** (right) | 45° — upper-right |
| 45° — upper-right | **ACTION2** (down) | 90° — 3 o'clock (right) |
| 90° — 3 o'clock (right) | **ACTION2** (down) | 135° — lower-right |
| 135° — lower-right | **ACTION3** (left) | 180° — 6 o'clock (below) |
| 180° — 6 o'clock (below) | **ACTION3** (left) | 225° — lower-left |
| 225° — lower-left | **ACTION1** (up) | 270° — 9 o'clock (left) |
| 270° — 9 o'clock (left) | **ACTION1** (up) | 315° — upper-left |
| 315° — upper-left | **ACTION4** (right) | 0° — 12 o'clock (above) |

> **Tip:** To reach the 6 o'clock (below) position from the initial 12 o'clock, apply 4 clockwise steps: ACTION4 → ACTION2 → ACTION2 → ACTION3.

### 2. Dye Color
- Shown by the container's interior color.
- **ACTION6** at a color swatch's (col, row) → switches to that color.
- Available swatches are visible in the **top green strip**.

### 3. Target Object
- The central multi-cell block that must be painted.
- Initially unpainted (black or a base color).
- Cells stay painted once colored; can be repainted by applying a different dye.

---

## Objective

Paint the **target object** so it exactly matches the **reference pattern** shown in the top-left yellow-bordered panel.

---

## Dyeing Mechanism

When ACTION5 is pressed:
- The container extends from its current orbital position toward the target center.
- Every cell of the target that the container **overlaps** gets painted with the current dye color.
- The overlap region depends on the container's orbital angle:
  - Container **directly above** (open side facing down) → paints the **top portion** of the target.
  - Container **directly below** (open side facing up) → paints the **bottom portion**.
  - Container **directly left** (open side facing right) → paints the **left portion**.
  - Container **directly right** (open side facing left) → paints the **right portion**.
  - Container at a **45° diagonal** → paints a **corner region** of the target.

---

## Step-by-Step Strategy

1. **Read the reference** (top-left, yellow border): identify which cells of the target should be which color.
2. **Identify the current dye color** (container interior color).
3. For each region of the target that needs a specific color:
   a. Click the matching swatch (**ACTION6** at the swatch position) to select the correct dye.
   b. Rotate the container (**ACTION3** / **ACTION4**) until its open side points directly at that region of the target.
   c. Press **ACTION5** to paint.
4. Repeat until the target matches the reference.

---

## Key Rules

1. The container's **open side always faces the target center** regardless of orbital angle.
2. ACTION5 **only paints cells where the container overlaps the target** — position the container carefully before applying dye.
3. Cells **can be repainted** — if you apply the wrong color, correct it by selecting the right dye and applying again.
4. The color swatch panel is in the **green strip at the very top** of the screen. Identify the (col, row) of each swatch before clicking.
5. Rotating ACTION3 or ACTION4 multiple times moves through discrete orbital positions (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°).
