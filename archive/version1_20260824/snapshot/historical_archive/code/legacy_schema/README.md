# Archived single-layer Schema code

This directory contains the historical `Concept` / `Relation` implementation
that preceded the active `MemoryRecord -> SchemaNode` architecture.

The files were moved out of `socialclaw/` on 2026-07-24 so current runners
cannot accidentally import the obsolete graph. They are retained for source
history only and are not installed as part of the `socialclaw` package.

`arc_agi3_parser.py` is the complete historical parser. The active
`socialclaw/schema/arc_agi3_parser.py` now contains only the three grid helpers
still used by current runners: object extraction, color naming, and grid diff.
