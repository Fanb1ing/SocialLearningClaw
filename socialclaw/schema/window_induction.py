from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..memory.assets import ContentAddressedArtifactStore, MemoryArtifactRef
from ..memory.models import MemoryRecord, utc_now
from .layered_graph import LayeredSchemaGraph
from .layered_storage import LayeredSchemaStorage
from .node import MemoryIndex, SchemaNode


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _effect_class(record: MemoryRecord) -> str:
    metadata = record.metadata
    status = str(metadata.get("environment_status", ""))
    if status == "WIN":
        return "win"
    if status == "GAME_OVER":
        return "game_over"
    if int(metadata.get("level_delta", 0) or 0) > 0:
        return "level_completion"
    return "effect" if metadata.get("task_state_changed") else "no_effect"


class ProposalOperation(str, Enum):
    CREATE = "create"
    SUPPORT = "support"
    REVISE = "revise"
    CONTRADICT = "contradict"
    SKIP = "skip"


@dataclass(frozen=True)
class VisualTransitionProfile:
    semantic_key: str
    base_key: str
    game_id: str
    action_name: str
    target_role: str
    effect_class: str
    region: str
    change_scale: str
    changed_cells: int
    level: int
    trigger: str
    expectation: str
    action_text: str
    pre_grid_artifact: Optional[Dict[str, Any]] = None
    post_grid_artifact: Optional[Dict[str, Any]] = None
    pre_view_artifact: Optional[Dict[str, Any]] = None
    post_view_artifact: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class KeyframeSelection:
    memory_id: str
    reasons: List[str]
    profile: VisualTransitionProfile

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["profile"] = asdict(self.profile)
        return value


@dataclass(frozen=True)
class SchemaProposal:
    operation: ProposalOperation
    semantic_key: str
    evidence_memory_ids: List[str]
    target_schema_id: str = ""
    trigger: str = ""
    action_sequence: List[str] = field(default_factory=list)
    expectation: str = ""
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["operation"] = self.operation.value
        return value


@dataclass(frozen=True)
class ProposalDecision:
    accepted: bool
    reason: str


class TransitionProfiler(Protocol):
    """Benchmark adapter boundary for extracting task-specific semantics."""

    def profile(self, record: MemoryRecord) -> VisualTransitionProfile:
        ...


class WindowProposalGenerator(Protocol):
    """Generator boundary shared by deterministic and future LLM backends."""

    def propose(
        self,
        records: Sequence[MemoryRecord],
        graph: LayeredSchemaGraph,
        *,
        keyframes: Sequence[KeyframeSelection] = (),
    ) -> List[SchemaProposal]:
        ...


class ARCVisualTransitionProfiler:
    """Extract compact semantics from stored ARC pre/post visual evidence.

    The profiler reads only trajectory-owned artifacts. It never imports a
    game implementation or a Gold schema, so the learned path remains isolated.
    """

    def __init__(self) -> None:
        self._grid_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def profile(self, record: MemoryRecord) -> VisualTransitionProfile:
        metadata = record.metadata
        game_id = str(metadata.get("game_id", record.task))
        action_name = str(metadata.get("action_name", ""))
        target_role = str(metadata.get("target_role", ""))
        level = int(metadata.get("level", 0) or 0)
        effect = _effect_class(record)
        pre_grid_ref = self._artifact(metadata.get("pre_artifacts"), "environment_state")
        post_grid_ref = self._artifact(metadata.get("post_artifacts"), "environment_state")
        pre_view_ref = self._artifact(metadata.get("pre_artifacts"), "agent_view")
        post_view_ref = self._artifact(metadata.get("post_artifacts"), "agent_view")
        changed_cells = int(metadata.get("task_changed_cells") or 0)
        bbox: Optional[Tuple[int, int, int, int]] = None
        if pre_grid_ref and post_grid_ref:
            pre = self._load_grid(record, pre_grid_ref)
            post = self._load_grid(record, post_grid_ref)
            if pre.shape == post.shape:
                task_rows = min(pre.shape[0], 63) if pre.shape[0] == 64 else pre.shape[0]
                coordinates = np.argwhere(pre[:task_rows] != post[:task_rows])
                changed_cells = int(coordinates.shape[0])
                if coordinates.size:
                    row0, col0 = coordinates.min(axis=0)
                    row1, col1 = coordinates.max(axis=0)
                    bbox = (int(row0), int(col0), int(row1), int(col1))
        region = self._region(game_id, effect, bbox)
        scale = self._scale(changed_cells)
        role_key = target_role or "none"
        semantic_key = "|".join(
            (game_id, action_name, role_key, effect, region, scale)
        )
        base_key = "|".join((game_id, action_name, role_key))
        role_text = f" on {target_role}" if target_role else ""
        trigger = self._trigger(game_id, action_name, role_text, effect, region)
        expectation = self._expectation(effect, region, scale, changed_cells)
        return VisualTransitionProfile(
            semantic_key=semantic_key,
            base_key=base_key,
            game_id=game_id,
            action_name=action_name,
            target_role=target_role,
            effect_class=effect,
            region=region,
            change_scale=scale,
            changed_cells=changed_cells,
            level=level,
            trigger=trigger,
            expectation=expectation,
            action_text=action_name + role_text,
            pre_grid_artifact=pre_grid_ref,
            post_grid_artifact=post_grid_ref,
            pre_view_artifact=pre_view_ref,
            post_view_artifact=post_view_ref,
        )

    @staticmethod
    def _artifact(values: Any, role: str) -> Optional[Dict[str, Any]]:
        for value in values or []:
            if value.get("role") == role:
                return dict(value)
        return None

    def _load_grid(self, record: MemoryRecord, value: Dict[str, Any]) -> np.ndarray:
        corpus_root = str(record.metadata.get("corpus_root", ""))
        reference = MemoryArtifactRef.from_dict(value)
        key = (corpus_root, reference.artifact_id)
        if key not in self._grid_cache:
            store = ContentAddressedArtifactStore(Path(corpus_root) / "assets")
            self._grid_cache[key] = store.load_grid(reference)
        return self._grid_cache[key]

    @staticmethod
    def _scale(changed_cells: int) -> str:
        if changed_cells <= 0:
            return "none"
        if changed_cells <= 25:
            return "local"
        if changed_cells <= 150:
            return "medium"
        return "global"

    @staticmethod
    def _region(
        game_id: str,
        effect: str,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> str:
        if effect in {"level_completion", "win", "game_over"}:
            return "level_transition"
        if bbox is None:
            return "none"
        row0, col0, row1, col1 = bbox
        game = game_id.lower()
        if game.startswith("cd82"):
            if row0 >= 33 and row1 <= 45 and col0 >= 26 and col1 <= 38:
                return "canvas"
            if row1 <= 13:
                return "palette_or_target"
            return "tool_area"
        if game.startswith("sk48"):
            return "chain_playfield"
        if game.startswith("tu93"):
            return "maze_board"
        vertical = "top" if row1 < 21 else "bottom" if row0 >= 42 else "middle"
        horizontal = "left" if col1 < 21 else "right" if col0 >= 42 else "center"
        return f"{vertical}_{horizontal}"

    @staticmethod
    def _trigger(
        game_id: str,
        action_name: str,
        role_text: str,
        effect: str,
        region: str,
    ) -> str:
        if effect == "no_effect":
            condition = "in an observed blocked or already-satisfied context"
        else:
            condition = "in an observed applicable context"
        return f"In {game_id}, when {action_name}{role_text} is used {condition}"

    @staticmethod
    def _expectation(effect: str, region: str, scale: str, changed_cells: int) -> str:
        if effect == "no_effect":
            return "The task grid remains unchanged; UI counters may still advance"
        if effect == "level_completion":
            return "The current level completes and the next level becomes visible"
        if effect == "win":
            return "The final level completes and the environment reports WIN"
        if effect == "game_over":
            return "A terminal failure condition is reached and the environment reports GAME_OVER"
        return (
            f"The {region} changes at {scale} scale "
            f"(representative transition changed {changed_cells} task cells)"
        )


class DeterministicKeyframeSelector:
    """Keep semantically new or terminal transitions, not every repeated frame."""

    def __init__(self) -> None:
        self._seen_semantics: set[str] = set()

    def select(
        self,
        records: Sequence[MemoryRecord],
        profiles: Mapping[str, VisualTransitionProfile],
    ) -> List[KeyframeSelection]:
        selected: List[KeyframeSelection] = []
        for record in records:
            profile = profiles[record.id]
            reasons: List[str] = []
            if profile.semantic_key not in self._seen_semantics:
                reasons.append("first_semantic_signature")
                self._seen_semantics.add(profile.semantic_key)
            if profile.effect_class in {"level_completion", "win", "game_over"}:
                reasons.append("terminal_or_level_boundary")
            if reasons:
                selected.append(KeyframeSelection(record.id, reasons, profile))
        return selected


class ProposalValidator:
    """Reject proposals that are ungrounded, out of scope, or inconsistent."""

    def __init__(self, memories: Mapping[str, MemoryRecord]) -> None:
        self.memories = memories

    def validate(
        self, proposal: SchemaProposal, graph: LayeredSchemaGraph
    ) -> ProposalDecision:
        if proposal.operation == ProposalOperation.SKIP:
            return ProposalDecision(True, "explicit_skip")
        if not proposal.evidence_memory_ids:
            return ProposalDecision(False, "missing_evidence")
        evidence: List[MemoryRecord] = []
        for memory_id in proposal.evidence_memory_ids:
            record = self.memories.get(memory_id)
            if record is None:
                return ProposalDecision(False, f"unknown_memory:{memory_id}")
            if record.metadata.get("memory_scope") != "transition":
                return ProposalDecision(False, f"non_transition_evidence:{memory_id}")
            evidence.append(record)
        games = {str(item.metadata.get("game_id")) for item in evidence}
        if len(games) != 1 or games != {str(proposal.metadata.get("game_id"))}:
            return ProposalDecision(False, "cross_game_or_scope_mismatch")
        expected_action = str(proposal.metadata.get("action_name", ""))
        actions = {str(item.metadata.get("action_name", "")) for item in evidence}
        if expected_action and actions != {expected_action}:
            return ProposalDecision(False, "action_scope_mismatch")
        if proposal.operation == ProposalOperation.CREATE:
            if proposal.target_schema_id:
                return ProposalDecision(False, "create_has_target")
            if not proposal.trigger or not proposal.action_sequence or not proposal.expectation:
                return ProposalDecision(False, "incomplete_rule")
            if any(
                node.metadata.get("semantic_key") == proposal.semantic_key
                for node in graph.list(include_inactive=True)
            ):
                return ProposalDecision(False, "duplicate_semantic_key")
        else:
            target = graph.get(proposal.target_schema_id)
            if target is None:
                return ProposalDecision(False, "unknown_target_schema")
            if target.metadata.get("game_id") not in games:
                return ProposalDecision(False, "target_game_mismatch")
            if proposal.operation == ProposalOperation.SUPPORT:
                if target.metadata.get("semantic_key") != proposal.semantic_key:
                    return ProposalDecision(False, "support_semantic_mismatch")
        return ProposalDecision(True, "grounded")


class DeterministicWindowProposalGenerator:
    """Offline proposal generator; an LLM version can implement the same contract."""

    def __init__(
        self,
        global_support: Mapping[str, int],
        profiles: Mapping[str, VisualTransitionProfile],
        *,
        min_support: int = 2,
    ) -> None:
        self.global_support = global_support
        self.profiles = profiles
        self.min_support = min_support

    def propose(
        self,
        records: Sequence[MemoryRecord],
        graph: LayeredSchemaGraph,
        *,
        keyframes: Sequence[KeyframeSelection] = (),
    ) -> List[SchemaProposal]:
        groups: Dict[str, List[MemoryRecord]] = defaultdict(list)
        for record in records:
            groups[self.profiles[record.id].semantic_key].append(record)
        proposals: List[SchemaProposal] = []
        planned_pair_revisions: set[Tuple[str, str]] = set()
        for semantic_key, evidence in sorted(groups.items()):
            profile = self.profiles[evidence[0].id]
            memory_ids = sorted(item.id for item in evidence)
            keyframe_ids = sorted(
                item.memory_id
                for item in keyframes
                if item.profile.semantic_key == semantic_key
            )
            existing = self._by_semantic_key(graph, semantic_key)
            common = {
                "game_id": profile.game_id,
                "action_name": profile.action_name,
                "target_role": profile.target_role,
                "base_key": profile.base_key,
                "effect_class": profile.effect_class,
                "region": profile.region,
                "change_scale": profile.change_scale,
                "level": profile.level,
                "keyframe_memory_ids": keyframe_ids,
            }
            if self.global_support[semantic_key] < self.min_support:
                proposals.append(
                    SchemaProposal(
                        operation=ProposalOperation.SKIP,
                        semantic_key=semantic_key,
                        evidence_memory_ids=memory_ids,
                        rationale="below_global_min_support",
                        metadata=common,
                    )
                )
                continue
            if existing is not None:
                proposals.append(
                    SchemaProposal(
                        operation=ProposalOperation.SUPPORT,
                        semantic_key=semantic_key,
                        evidence_memory_ids=memory_ids,
                        target_schema_id=existing.index,
                        rationale="additional_same_semantics_evidence",
                        metadata=common,
                    )
                )
                continue
            for paired in self._paired_nodes(graph, profile):
                paired_classes = set(paired.metadata.get("paired_effect_classes", []))
                pair_marker = (paired.index, profile.effect_class)
                if (
                    profile.effect_class not in paired_classes
                    and pair_marker not in planned_pair_revisions
                ):
                    qualification = "; other observed contexts produce different outcomes"
                    proposals.append(
                        SchemaProposal(
                            operation=ProposalOperation.REVISE,
                            semantic_key=str(paired.metadata.get("semantic_key")),
                            evidence_memory_ids=memory_ids,
                            target_schema_id=paired.index,
                            trigger=(
                                paired.trigger
                                if qualification in paired.trigger
                                else paired.trigger + qualification
                            ),
                            expectation=paired.expectation,
                            rationale="paired_outcome_requires_conditional_scope",
                            metadata={**common, "evidence_polarity": "negative"},
                        )
                    )
                    planned_pair_revisions.add(pair_marker)
            proposals.append(
                SchemaProposal(
                    operation=ProposalOperation.CREATE,
                    semantic_key=semantic_key,
                    evidence_memory_ids=memory_ids,
                    trigger=profile.trigger,
                    action_sequence=[profile.action_text],
                    expectation=profile.expectation,
                    rationale="first_grounded_semantic_group",
                    metadata=common,
                )
            )
        return proposals

    @staticmethod
    def _by_semantic_key(
        graph: LayeredSchemaGraph, semantic_key: str
    ) -> Optional[SchemaNode]:
        return next(
            (
                node
                for node in graph.list(include_inactive=True)
                if node.metadata.get("semantic_key") == semantic_key
            ),
            None,
        )

    @staticmethod
    def _paired_nodes(
        graph: LayeredSchemaGraph, profile: VisualTransitionProfile
    ) -> List[SchemaNode]:
        return [
            node
            for node in graph.list(include_inactive=True)
            if node.metadata.get("base_key") == profile.base_key
            and node.metadata.get("effect_class") != profile.effect_class
            and "no_effect"
            in {str(node.metadata.get("effect_class")), profile.effect_class}
        ]


class ProposalApplier:
    def apply(self, proposal: SchemaProposal, graph: LayeredSchemaGraph) -> str:
        evidence = sorted(set(proposal.evidence_memory_ids))
        if proposal.operation == ProposalOperation.SKIP:
            return ""
        if proposal.operation == ProposalOperation.CREATE:
            node = SchemaNode.from_rule(
                level=3,
                trigger=proposal.trigger,
                action_sequence=proposal.action_sequence,
                expectation=proposal.expectation,
                source_memory_id=evidence[0],
                reliability_weight=min(0.9, 0.5 + 0.04 * len(evidence)),
            )
            node.index = _stable_id("schema", {"semantic_key": proposal.semantic_key})
            node.memory_index = MemoryIndex(source=evidence)
            node.metadata.update(
                {
                    **proposal.metadata,
                    "algorithm": "semantic_window_v1",
                    "semantic_key": proposal.semantic_key,
                    "support_count": len(evidence),
                    "level_scope": sorted({int(proposal.metadata.get("level", 0))}),
                    "schema_kind": (
                        "state_transition"
                        if proposal.metadata.get("effect_class")
                        in {"level_completion", "win", "game_over"}
                        else "constraint"
                        if proposal.metadata.get("effect_class") == "no_effect"
                        else "action_effect"
                    ),
                    "paired_effect_classes": [],
                }
            )
            graph.add(node)
            return node.index
        node = graph.get(proposal.target_schema_id)
        if node is None:
            raise KeyError(proposal.target_schema_id)
        if proposal.operation == ProposalOperation.SUPPORT:
            node.memory_index.source = sorted({*node.memory_index.source, *evidence})
            node.metadata["support_count"] = len(node.memory_index.source)
            levels = {*node.metadata.get("level_scope", []), int(proposal.metadata.get("level", 0))}
            node.metadata["level_scope"] = sorted(levels)
            keyframes = {
                *node.metadata.get("keyframe_memory_ids", []),
                *proposal.metadata.get("keyframe_memory_ids", []),
            }
            node.metadata["keyframe_memory_ids"] = sorted(keyframes)
            node.reliability_weight = min(0.95, 0.5 + 0.04 * len(node.memory_index.source))
        elif proposal.operation == ProposalOperation.REVISE:
            node.trigger = proposal.trigger or node.trigger
            node.expectation = proposal.expectation or node.expectation
            node.description = self._description(node)
            if proposal.metadata.get("evidence_polarity") == "negative":
                node.memory_index.negative = sorted({*node.memory_index.negative, *evidence})
            else:
                node.memory_index.source = sorted({*node.memory_index.source, *evidence})
            paired = set(node.metadata.get("paired_effect_classes", []))
            paired.add(str(proposal.metadata.get("effect_class")))
            node.metadata["paired_effect_classes"] = sorted(paired)
            negative_keyframes = {
                *node.metadata.get("negative_keyframe_memory_ids", []),
                *proposal.metadata.get("keyframe_memory_ids", []),
            }
            node.metadata["negative_keyframe_memory_ids"] = sorted(negative_keyframes)
        elif proposal.operation == ProposalOperation.CONTRADICT:
            node.memory_index.negative = sorted({*node.memory_index.negative, *evidence})
            node.reliability_weight = max(0.05, node.reliability_weight - 0.05 * len(evidence))
        node.updated_at = utc_now()
        return node.index

    @staticmethod
    def _description(node: SchemaNode) -> str:
        return (
            f"[Context/Perception: {node.trigger}] "
            f"[Action/Execution: {' -> '.join(node.action_sequence)}] "
            f"[Expectation: {node.expectation}]"
        )


class WindowSchemaInductionScheduler:
    """Process trajectory windows and persist a complete proposal audit trail."""

    def __init__(
        self,
        *,
        profiler: Optional[TransitionProfiler] = None,
        min_support: int = 2,
        generator_factory: Optional[
            Callable[
                [Mapping[str, int], Mapping[str, VisualTransitionProfile], int],
                WindowProposalGenerator,
            ]
        ] = None,
    ) -> None:
        self.profiler = profiler or ARCVisualTransitionProfiler()
        self.min_support = min_support
        self.generator_factory = generator_factory

    def run(
        self,
        memories: Iterable[MemoryRecord],
        *,
        initial_graph: Optional[LayeredSchemaGraph] = None,
    ) -> Tuple[LayeredSchemaGraph, Dict[str, Any], List[Dict[str, Any]], List[KeyframeSelection]]:
        records = list(memories)
        memory_by_id = {record.id: record for record in records}
        transitions = {
            record.id: record
            for record in records
            if record.metadata.get("memory_scope") == "transition"
        }
        profiles = {memory_id: self.profiler.profile(record) for memory_id, record in transitions.items()}
        global_support = Counter(profile.semantic_key for profile in profiles.values())
        windows = sorted(
            (record for record in records if record.metadata.get("memory_scope") == "window_summary"),
            key=lambda item: (
                item.task,
                str(item.metadata.get("episode_id", "")),
                int(item.metadata.get("start_step", 0)),
            ),
        )
        graph = initial_graph or LayeredSchemaGraph()
        selector = DeterministicKeyframeSelector()
        if self.generator_factory:
            generator = self.generator_factory(global_support, profiles, self.min_support)
        else:
            generator = DeterministicWindowProposalGenerator(
                global_support, profiles, min_support=self.min_support
            )
        validator = ProposalValidator(memory_by_id)
        applier = ProposalApplier()
        audit: List[Dict[str, Any]] = []
        keyframes: List[KeyframeSelection] = []
        for window in windows:
            source = [
                transitions[memory_id]
                for memory_id in window.metadata.get("source_memory_ids", [])
                if memory_id in transitions
            ]
            window_keyframes = selector.select(source, profiles)
            keyframes.extend(window_keyframes)
            for proposal in generator.propose(source, graph, keyframes=window_keyframes):
                decision = validator.validate(proposal, graph)
                schema_id = applier.apply(proposal, graph) if decision.accepted else ""
                audit.append(
                    {
                        "index": len(audit),
                        "window_memory_id": window.id,
                        "proposal": proposal.to_dict(),
                        "accepted": decision.accepted,
                        "decision_reason": decision.reason,
                        "affected_schema_id": schema_id,
                    }
                )
        graph.validate(memory_ids=set(memory_by_id))
        for node in graph.list(include_inactive=True):
            keyframe_ids = {
                *node.metadata.get("keyframe_memory_ids", []),
                *node.metadata.get("negative_keyframe_memory_ids", []),
            }
            invalid_keyframes = keyframe_ids - set(transitions)
            if invalid_keyframes:
                raise ValueError(
                    f"Schema {node.index} cites invalid keyframes: {sorted(invalid_keyframes)}"
                )
        operation_counts = Counter(item["proposal"]["operation"] for item in audit)
        source_evidence = {
            memory_id
            for node in graph.list(include_inactive=True)
            for memory_id in node.memory_index.source
        }
        negative_evidence = {
            memory_id
            for node in graph.list(include_inactive=True)
            for memory_id in node.memory_index.negative
        }
        report = {
            "format_version": 1,
            "algorithm": "semantic_window_v1",
            "memory_count": len(records),
            "transition_count": len(transitions),
            "window_count": len(windows),
            "keyframe_count": len(keyframes),
            "schema_count": len(graph.list(include_inactive=True)),
            "schema_source_transition_count": len(source_evidence),
            "schema_negative_transition_count": len(negative_evidence),
            "uncited_transition_count": len(set(transitions) - source_evidence),
            "schema_counts_by_game": dict(
                sorted(Counter(str(node.metadata.get("game_id")) for node in graph.list()).items())
            ),
            "proposal_counts": dict(sorted(operation_counts.items())),
            "accepted_proposal_count": sum(item["accepted"] for item in audit),
            "applied_proposal_count": sum(
                item["accepted"] and item["proposal"]["operation"] != "skip"
                for item in audit
            ),
            "rejected_proposal_count": sum(not item["accepted"] for item in audit),
            "network_calls": 0,
            "gold_schema_reads": 0,
        }
        return graph, report, audit, keyframes


def run_window_induction(memory_path: str | Path, output_root: str | Path) -> Dict[str, Any]:
    from ..memory import JsonMemoryStore

    memories = JsonMemoryStore(memory_path).list()
    graph, report, audit, keyframes = WindowSchemaInductionScheduler().run(memories)
    memory_by_id = {record.id: record for record in memories}
    verified_artifacts: set[Tuple[str, str]] = set()
    for keyframe in keyframes:
        record = memory_by_id[keyframe.memory_id]
        corpus_root = str(record.metadata.get("corpus_root", ""))
        store = ContentAddressedArtifactStore(Path(corpus_root) / "assets")
        for value in (
            keyframe.profile.pre_grid_artifact,
            keyframe.profile.post_grid_artifact,
            keyframe.profile.pre_view_artifact,
            keyframe.profile.post_view_artifact,
        ):
            if value:
                reference = MemoryArtifactRef.from_dict(value)
                store.verify(reference)
                verified_artifacts.add((corpus_root, reference.artifact_id))
    report["verified_keyframe_artifact_count"] = len(verified_artifacts)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    LayeredSchemaStorage(output / "schema.json").save(graph)
    for name, value in (
        ("report.json", report),
        ("audit.json", audit),
        ("keyframes.json", [item.to_dict() for item in keyframes]),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report
