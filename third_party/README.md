# Third-party Sources

`arc_agi3_games/` contains downloaded ARC Prize Foundation game environment
sources and metadata. The source files retain their MIT license headers. The
experiment wrapper loads this directory in SDK offline mode so formal runs use
the pinned local game versions rather than silently downloading newer code.
`arc_agi3_games/inventory.json` records the 25 full game IDs and SHA-256 hashes
of every metadata/source pair. Host-specific `local_dir` fields are removed;
the SDK reconstructs them while scanning the configured directory.

Refresh the local inventory with:

```bash
.venv/bin/python scripts/download_arc_games.py
```

To verify and regenerate the inventory without contacting the API:

```bash
.venv/bin/python scripts/download_arc_games.py --normalize-only
```

These files define benchmark environments; they are not SocialLearningClaw agent implementations.
