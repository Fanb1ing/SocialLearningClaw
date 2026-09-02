"""Tycho: active abstraction for interactive ARC-AGI-3 environments."""

import os as _os

# ARCBaseGame's default RESET semantics allow a RESET at action_count==0 to restart the
# whole game from level 0. Tycho's harness contract treats RESET as a same-level recovery
# control, matching the competition wrapper behavior and preventing score/regression drift.
_os.environ["ONLY_RESET_LEVELS"] = "true"
