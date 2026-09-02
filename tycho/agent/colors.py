"""Shared color palette description, used by BOTH the actor (agent.py) and the world-model
builder (builder.py) so they refer to grid colors the same way (one wrote about "color 5", the
other had no mapping and had to re-derive it — they must speak the same language)."""

COLORMAP = (
    "Color palette (cell value → rendered color): 0=white, 1=light-grey, 2=grey, 3=dark-grey, "
    "4=charcoal, 5=black, 6=magenta, 7=pink, 8=red, 9=blue, a=cyan, b=yellow, c=orange, "
    "d=maroon, e=green, f=purple. Treat them as opaque IDs; the names are just for reference."
)
