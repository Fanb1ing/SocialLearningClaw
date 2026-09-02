import numpy as np

from tycho.harness.animation_evidence import (
    AnimationEvidenceConfig,
    analyze_frame_sequence,
)


def _blank():
    return np.zeros((64, 64), dtype=np.int16)


def test_suppresses_simple_local_motion():
    frames = []
    for c in range(4):
        g = _blank()
        g[30:32, 20 + c:22 + c] = 7
        frames.append(g)

    decision = analyze_frame_sequence(frames)

    assert not decision.surface
    assert decision.category == "local_motion"
    assert decision.unique_frame_count == 4


def test_surfaces_viewport_or_reveal_transition():
    frames = []
    base = np.zeros((64, 64), dtype=np.int16)
    base[10:45, 20:44] = 3
    for i in range(6):
        g = np.zeros((64, 64), dtype=np.int16)
        src = np.roll(base, shift=-i * 4, axis=0)
        g[:, :] = src
        g[: i + 1, :] = 9  # new border content enters during the shift
        frames.append(g)

    decision = analyze_frame_sequence(frames)

    assert decision.surface
    assert decision.category in {"viewport_or_reveal", "eventful_transition", "nonredundant_intermediate"}
    assert "border changes suggest viewport/hidden-world reveal" in decision.reasons
    assert decision.selected_original_indices[0] == 0
    assert decision.selected_original_indices[-1] == len(frames) - 1


def test_surfaces_retracting_transient_when_endpoint_hides_motion():
    base = _blank()
    extended = base.copy()
    extended[10:20, 10:22] = 12
    extended[20:30, 28:40] = 13
    frames = [base, extended, base.copy()]

    decision = analyze_frame_sequence(frames)

    assert decision.surface
    assert decision.category == "retracting_transient"
    assert "intermediate frames changed much more than the endpoint delta" in decision.reasons
    assert decision.features["endpoint_cells"] == 0
    assert decision.features["path_cells"] > 0


def test_exact_dedupe_can_make_endpoint_sufficient():
    g0 = _blank()
    g1 = _blank()
    g1[4, 4] = 2

    decision = analyze_frame_sequence([g0, g0.copy(), g1, g1.copy()])

    assert not decision.surface
    assert decision.category == "endpoint_sufficient"
    assert decision.original_frame_count == 4
    assert decision.unique_frame_count == 2


def test_keyframe_budget_is_respected():
    frames = []
    for i in range(12):
        g = _blank()
        g[:, : i + 1] = 4
        g[i:64:8, :] = 8
        frames.append(g)

    decision = analyze_frame_sequence(frames, AnimationEvidenceConfig(max_keyframes=5))

    assert decision.surface
    assert len(decision.selected_original_indices) <= 5
    assert 0 in decision.selected_original_indices
    assert len(frames) - 1 in decision.selected_original_indices
