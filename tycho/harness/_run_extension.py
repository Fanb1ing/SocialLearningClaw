"""SocialLearningClaw's narrow use of Tycho's documented runner extension."""

APPROACH_MODULES = {
    "tycho_efps": "socialclaw.v3.agent",
}

# These files affect policy and workspace contents, so Tycho's immutable run
# specification must hash them alongside its own actor, prompts and tooling.
POLICY_PATHS = (
    "socialclaw/v3",
)


def execution_extra_sources(games: dict) -> dict[str, str]:
    """Bind each local game implementation to the immutable Tycho run spec."""
    import os

    from socialclaw.dataset.arc_agi3 import arc_environment_fingerprint

    environments_dir = os.environ.get("TYCHO_ENVIRONMENTS_DIR") or None
    return {
        f"environment/{game_id}": arc_environment_fingerprint(
            game_id,
            environments_dir=environments_dir,
        )
        for game_id in sorted(games)
    }
