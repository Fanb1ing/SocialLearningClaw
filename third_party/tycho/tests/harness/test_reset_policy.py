import os


def test_tycho_forces_arc_level_reset_only():
    import tycho  # noqa: F401

    assert os.environ["ONLY_RESET_LEVELS"] == "true"


def test_official_scorecard_charges_in_play_reset_but_not_initialization():
    from arc_agi.scorecard import Card

    card = Card(game_id="test-game")
    card.inc_play_count("guid")
    assert card.actions == [0]
    assert card.resets == [0]

    card.inc_reset_count("guid")
    assert card.actions == [1]
    assert card.resets == [1]
