# SK48 Game Rules

## Screen Layout

The screen has **two distinct regions**:

- **Upper area**: the active play field. This contains the yellow zone with the square objects to be threaded, the red-green tracks, and the pink-black frame with the rod.
- **Bottom area**: a **reference diagram only**. It shows the target arrangement — the order and positions in which the squares must end up on the rod after threading. This diagram does **not** interact with the game; it is purely visual reference.

> **Important**: The actual squares you need to thread are the ones in the **upper yellow zone**. The bottom diagram is an example of the desired final state, not a playable area.

---

## Controllable Elements

There are exactly **two** things you can control:

### 1. Pink-Black Frame (the carrier)
- Moves **vertically only**, along the red-green intertwined tracks (which are in the upper play area).
- Carries the red-blue rod with it at all times.
- **ACTION1** → move the frame **UP** (row decreases by 1)
- **ACTION2** → move the frame **DOWN** (row increases by 1)

### 2. Red-Blue Intertwined Rod (the skewer)
- Attached to the pink-black frame; moves with the frame vertically.
- Can extend or retract **horizontally only**.
- **ACTION3** → extend/retract the rod **LEFTWARD**
- **ACTION4** → extend/retract the rod **RIGHTWARD**

Everything else in the scene (tracks, yellow zone, square objects, bottom reference diagram) is part of the environment and cannot be directly controlled.

---

## Environment (upper play area)

- **Red-green intertwined tracks**: the vertical rail in the upper area along which the pink-black frame travels.
- **Yellow zone**: the rectangular region in the upper area that contains the square target objects.
- **Square objects**: located inside the yellow zone. These are the targets that must be threaded onto the rod one by one.

---

## Objective

Thread **all square objects** (from the upper yellow zone) onto the red-blue rod, one by one, in the exact arrangement shown in the **bottom reference diagram**.

---

## Threading Mechanism

A square object is successfully threaded when **both** conditions hold simultaneously:

1. **Position condition**: The square object is at the **outermost edge** of the yellow zone (i.e., at the boundary closest to the exterior of the yellow region).
2. **Rod condition**: The rod extends **from the interior side** (non-edge side) of the square **through** the square **toward the exterior edge side**.

**Direction rule**: the rod must travel from inside the yellow zone outward through the square — never from outside inward.

### Example
> A square is at the **rightmost edge** of the yellow zone (upper area).
> → The rod must extend from the **left side** of the square toward the **right** (ACTION4), passing through the square outward.

> A square is at the **leftmost edge** of the yellow zone (upper area).
> → The rod must extend from the **right side** of the square toward the **left** (ACTION3), passing through the square outward.

---

## Rod Pushing Behavior

When the rod extends in a direction and encounters a square object that is blocking its path, the rod **pushes that square** in the direction of extension. The square will move along until it either:
- Reaches the edge of the yellow zone (and can be threaded if the rod continues through it), or
- Reaches an obstacle that prevents further movement.

This push mechanic is important: you can reposition squares by extending the rod into them.

---

## Step-by-Step Strategy

1. **Look at the bottom reference diagram** to determine the target order of squares to thread.
2. **Focus on the upper yellow zone** to find where the actual square objects are located. Note each square's (col, row) position in the grid.
3. **Identify the next square** to thread (per the target order) and locate it in the upper yellow zone.
4. **Move the frame vertically** (ACTION1 / ACTION2) until the rod row aligns with that square's row.
5. **Check which edge of the yellow zone** the square is closest to (left or right boundary).
6. **Retract the rod** to the non-edge side of the square if needed (ACTION3 or ACTION4).
7. **Extend the rod through the square toward the edge** (ACTION4 for rightward threading, ACTION3 for leftward threading) to thread it.
8. **Repeat** for the next square in the target order.
