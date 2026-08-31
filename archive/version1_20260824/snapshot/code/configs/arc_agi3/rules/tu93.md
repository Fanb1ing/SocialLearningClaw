# TU93 Game Rules

## Screen Layout

The screen is a **64×64 grid** with a mostly **gray background**. The play area contains:

- **Red-black alternating path**: The winding corridor/maze that forms the only walkable route.
  - The path is made of alternating red and black cells in a checkerboard-like pattern.
  - Gray cells outside the path are **walls / non-walkable**.
- **Player character** (dark maroon 3×3 block): Moves along the path.
- **Yellow pixel** (inside the player block): Indicates the player's **facing direction** (the open side / notch).
- **Goal** (magenta/pink 3×3 block): The destination the player must reach.

---

## Controllable Elements

There is **one controllable object**: the player (3×3 dark maroon block).

- **ACTION1** → move player **UP** along the path
- **ACTION2** → move player **DOWN** along the path
- **ACTION3** → move player **LEFT** along the path
- **ACTION4** → move player **RIGHT** along the path

The player can only move along the **red-black path** — it cannot step onto gray cells.

---

## Objective

Navigate the player from its **start position** to the **magenta goal block** by moving along the red-black path, without stepping into danger.

---

## Enemies

Some levels contain **enemy characters** (same 3×3 block shape as the player but in a different color, with their own facing indicator):

- **Danger rule**: Do **NOT** move the player onto the cell that is directly in front of an enemy's open/notch side — the enemy will capture the player and the level fails (GAME_OVER).
- **Safe approach**: You **can** safely step onto any cell that is **behind** or **to the side** of an enemy. Doing so removes the enemy.
- To eliminate an enemy: approach from the side or rear, never from the front.

---

## Navigation Rules

1. The path is a **single connected winding corridor** — follow the red-black cells.
2. The player moves **one step at a time** along the path; it cannot jump over gaps.
3. If an action would move the player off the path (into gray), the move has **no effect**.
4. The yellow notch pixel shows which direction the player is currently facing; this changes as the player turns.

---

## Step-by-Step Strategy

1. **Locate the player** (dark maroon 3×3 block with yellow notch pixel) and the **goal** (magenta 3×3 block).
2. **Trace the red-black path** from the player's position to the goal.
3. **Plan the route**: identify the sequence of directional moves (up/down/left/right) needed to follow the path.
4. **Check for enemies** along the route: if an enemy blocks the path, approach from the side or rear to remove it.
5. **Execute moves** one at a time with ACTION1–4.

---

## Key Rules

1. Only move onto **red or black cells** (the path). Gray = wall, cannot be entered.
2. **Never step directly in front of an enemy's open notch** — that cell is lethal.
3. Approaching an enemy from the **side or back** eliminates it safely.
4. The **goal is reached** when the player moves onto the magenta block's position.