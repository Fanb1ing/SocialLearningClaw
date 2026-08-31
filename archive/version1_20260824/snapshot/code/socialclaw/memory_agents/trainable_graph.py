"""
Inference-time trainable graph memory baseline.

This is a lightweight reproduction of the graph-memory architecture in
"From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory"
without RL or model-parameter training. It keeps the paper's three explicit
layers: query nodes, canonical path nodes, and meta-cognition strategy nodes.
Edges are updated from downstream reward feedback and retrieved strategies are
injected as prompt context for later examples.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI


@dataclass
class QueryNode:
    id: str
    task: str
    outcome: bool
    reward: float
    trajectory: str = ""
    lesson: str = ""
    context: str = ""
    domain: str = ""
    path_id: str = ""
    created_at: str = ""


@dataclass
class PathNode:
    id: str
    states: List[str]
    summary: str
    domain: str = ""
    n_success: int = 0
    n_failure: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass
class MetaCognitionNode:
    id: str
    summary: str
    principles: List[str] = field(default_factory=list)
    confidence: float = 0.5
    evidence_count: int = 0
    reward_sum: float = 0.0
    domain: str = ""
    source_path_ids: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    usage_count: int = 0
    reward_sum: float = 0.0


class TrainableGraphMemory:
    """Three-layer graph memory with reward-weighted strategy retrieval."""

    def __init__(
        self,
        client: OpenAI,
        model_id: str,
        embedder,
        max_meta: int = 30,
        retrieve_k: int = 3,
        similar_query_k: int = 8,
        meta_merge_threshold: float = 0.84,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.embedder = embedder
        self.max_meta = max_meta
        self.retrieve_k = retrieve_k
        self.similar_query_k = similar_query_k
        self.meta_merge_threshold = meta_merge_threshold

        self.queries: List[QueryNode] = []
        self.paths: Dict[str, PathNode] = {}
        self.meta: Dict[str, MetaCognitionNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def get_memory_block(self, query: str = "", top_k: Optional[int] = None) -> str:
        nodes = self.retrieve(query=query, top_k=top_k)
        if not nodes:
            return ""
        lines = []
        for i, node in enumerate(nodes, 1):
            principles = " ".join(f"{p}" for p in node.principles[:3])
            body = f"{node.summary}"
            if principles:
                body += f" Principles: {principles}"
            lines.append(
                f"{i}. {body} "
                f"(confidence={node.confidence:.2f}, evidence={node.evidence_count})"
            )
        return (
            "=== Retrieved graph-memory meta-cognition strategies ===\n"
            + "\n".join(lines)
            + "\nUse these only when relevant; prefer current evidence over memory when they conflict.\n"
            + "========================================================\n"
        )

    def retrieve(self, query: str = "", top_k: Optional[int] = None) -> List[MetaCognitionNode]:
        if not self.meta:
            return []
        top_k = top_k or self.retrieve_k
        if not query:
            return sorted(
                self.meta.values(),
                key=lambda m: (m.confidence, m.evidence_count),
                reverse=True,
            )[:top_k]

        try:
            q_emb = self._encode(query)
        except Exception:
            return list(self.meta.values())[-top_k:]

        similar_queries = []
        for q in self.queries:
            emb = self.embeddings.get(q.id)
            if emb is None:
                continue
            similar_queries.append((float(emb @ q_emb), q))
        similar_queries.sort(key=lambda x: x[0], reverse=True)
        active_queries = similar_queries[: self.similar_query_k]

        scores: Dict[str, float] = {}
        if active_queries:
            for sim, qnode in active_queries:
                if sim <= 0.03:
                    continue
                qp = self._edge(qnode.id, qnode.path_id)
                if not qp:
                    continue
                for edge in self.edges.values():
                    if edge.source != qnode.path_id or edge.relation != "path_supports_meta":
                        continue
                    meta_node = self.meta.get(edge.target)
                    if not meta_node:
                        continue
                    edge_utility = max(0.05, edge.weight)
                    scores[meta_node.id] = scores.get(meta_node.id, 0.0) + (
                        sim * qp.weight * edge_utility * max(0.05, meta_node.confidence)
                    )

        if not scores:
            for node in self.meta.values():
                emb = self.embeddings.get(node.id)
                sim = float(emb @ q_emb) if emb is not None else 0.0
                scores[node.id] = sim * max(0.05, node.confidence)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.meta[mid] for mid, score in ranked[:top_k] if mid in self.meta and score > 0.0]

    def add_experience(
        self,
        *,
        task: str,
        outcome: bool,
        trajectory: str = "",
        lesson: str = "",
        context: str = "",
        domain: str = "",
    ) -> MetaCognitionNode:
        now = self._now()
        reward = 1.0 if outcome else -1.0
        qid = f"q_{uuid.uuid4().hex[:10]}"
        path = self._get_or_create_path(
            task=task,
            outcome=outcome,
            trajectory=trajectory,
            lesson=lesson,
            domain=domain,
        )
        if outcome:
            path.n_success += 1
        else:
            path.n_failure += 1
        path.updated_at = now

        qnode = QueryNode(
            id=qid,
            task=task[:2400],
            outcome=outcome,
            reward=reward,
            trajectory=trajectory[:2400],
            lesson=lesson[:1200],
            context=context[:1200],
            domain=domain,
            path_id=path.id,
            created_at=now,
        )
        self.queries.append(qnode)
        self.embeddings[qnode.id] = self._encode(self._query_text(qnode))

        meta_candidate = self._build_meta_cognition(
            task=task,
            outcome=outcome,
            trajectory=trajectory,
            lesson=lesson,
            context=context,
            path=path,
            domain=domain,
        )
        meta_node = self._merge_or_create_meta(meta_candidate, path_id=path.id, reward=reward, domain=domain)

        self._upsert_edge(qnode.id, path.id, "query_follows_path", reward=reward, base_weight=1.0)
        self._upsert_edge(path.id, meta_node.id, "path_supports_meta", reward=reward, base_weight=meta_node.confidence)
        self._trim()
        return meta_node

    def save(self, path: str | Path) -> None:
        payload = {
            "queries": [asdict(q) for q in self.queries],
            "paths": [asdict(p) for p in self.paths.values()],
            "meta": [asdict(m) for m in self.meta.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.queries = [QueryNode(**x) for x in data.get("queries", [])]
        self.paths = {x["id"]: PathNode(**x) for x in data.get("paths", [])}
        self.meta = {x["id"]: MetaCognitionNode(**x) for x in data.get("meta", [])}
        self.edges = {
            self._edge_key(x["source"], x["target"], x["relation"]): GraphEdge(**x)
            for x in data.get("edges", [])
        }
        self.embeddings = {}
        for q in self.queries:
            self.embeddings[q.id] = self._encode(self._query_text(q))
        for m in self.meta.values():
            self.embeddings[m.id] = self._encode(self._meta_text(m))

    # ── graph construction ───────────────────────────────────────────────────

    def _get_or_create_path(
        self,
        *,
        task: str,
        outcome: bool,
        trajectory: str,
        lesson: str,
        domain: str,
    ) -> PathNode:
        states = self._infer_states(task=task, outcome=outcome, trajectory=trajectory, lesson=lesson, domain=domain)
        signature = f"{domain or 'general'}:" + ">".join(states)
        pid = "p_" + uuid.uuid5(uuid.NAMESPACE_URL, signature).hex[:12]
        if pid in self.paths:
            return self.paths[pid]
        now = self._now()
        node = PathNode(
            id=pid,
            states=states,
            summary=" -> ".join(states),
            domain=domain,
            created_at=now,
            updated_at=now,
        )
        self.paths[pid] = node
        return node

    def _infer_states(self, *, task: str, outcome: bool, trajectory: str, lesson: str, domain: str) -> List[str]:
        text = f"{task}\n{trajectory}\n{lesson}".lower()
        if domain == "contextmath":
            states = ["ParseProblem"]
            if any(w in text for w in ["geometry", "triangle", "circle", "tetra", "angle", "area"]):
                states.append("IdentifyGeometryStructure")
            elif any(w in text for w in ["number", "mod", "integer", "divis", "prime"]):
                states.append("IdentifyNumberTheoryStructure")
            else:
                states.append("IdentifyAlgebraicStructure")
            if any(w in text for w in ["equation", "system", "solve", "let "]):
                states.append("SetUpEquation")
            states.extend(["Compute", "VerifyAnswer" if outcome else "MissedVerification", "Answer"])
            return states

        if domain == "intphys2":
            states = ["ObserveObjects"]
            if "permanence" in text:
                states.append("CheckObjectPermanence")
            elif "immutability" in text:
                states.append("CheckShapeImmutability")
            elif "continuity" in text:
                states.append("CheckMotionContinuity")
            elif "solidity" in text:
                states.append("CheckObjectSolidity")
            else:
                states.append("InferPhysicalConstraint")
            states.extend(["ComparePossibleImpossible", "Answer" if outcome else "BiasOrCueError"])
            return states

        if domain == "arc_agi3":
            states = ["ObserveGrid"]
            if "action6" in text or "click" in text:
                states.append("TestClickInteraction")
            elif any(a in text for a in ["action1", "action2", "action3", "action4", "move"]):
                states.append("TestMovement")
            else:
                states.append("ExploreAvailableAction")
            if "changed=true" in text or "grid changes" in text:
                states.append("UseActionEffect")
            elif "changed=false" in text or "no-effect" in text or "no effect" in text:
                states.append("AvoidNoEffectLoop")
            else:
                states.append("InferTransitionRule")
            states.extend(["AdvanceLevel" if outcome else "RevisePolicy", "AnswerAction"])
            return states

        return ["UnderstandTask", "Plan", "Execute", "Verify" if outcome else "DiagnoseFailure", "Answer"]

    def _build_meta_cognition(
        self,
        *,
        task: str,
        outcome: bool,
        trajectory: str,
        lesson: str,
        context: str,
        path: PathNode,
        domain: str,
    ) -> dict:
        existing = "\n".join(
            f"- {m.id}: {m.summary} (confidence={m.confidence:.2f})"
            for m in sorted(self.meta.values(), key=lambda x: x.confidence)[:8]
        ) or "(none)"
        prompt = (
            "You are building a reusable meta-cognition node for an LLM agent memory graph.\n"
            "Create or refine one concise strategy that can help future similar tasks.\n"
            "Keep it general, directly usable, and cautious. Do not include exact answer labels.\n"
            "Return ONLY valid JSON with keys: summary, principles, confidence.\n\n"
            f"Domain: {domain or 'general'}\n"
            f"Outcome: {'success' if outcome else 'failure'}\n"
            f"Canonical FSM path: {path.summary}\n"
            f"Task:\n{task[:1200]}\n\n"
            f"Trajectory/model output:\n{trajectory[:1200]}\n\n"
            f"Lesson/reward feedback:\n{lesson[:700]}\n\n"
            f"Context:\n{context[:500]}\n\n"
            f"Existing low-confidence strategies:\n{existing}\n\n"
            "JSON schema: {\"summary\": \"...\", \"principles\": [\"...\"], \"confidence\": 0.30-0.85}"
        )
        raw = self._call_llm(prompt, max_tokens=600)
        data = self._parse_json(raw)
        if not data:
            fallback = lesson or ("Successful strategy was useful." if outcome else "Avoid the observed failure mode.")
            data = {
                "summary": fallback[:420],
                "principles": [fallback[:260]],
                "confidence": 0.65 if outcome else 0.45,
            }
        return data

    def _merge_or_create_meta(self, data: dict, *, path_id: str, reward: float, domain: str) -> MetaCognitionNode:
        summary = str(data.get("summary", "")).strip()[:700]
        principles = [str(x).strip()[:360] for x in data.get("principles", []) if str(x).strip()][:5]
        if not summary and principles:
            summary = principles[0]
        if not summary:
            summary = "Use reward feedback to refine the next solution strategy."
        raw_conf = data.get("confidence", 0.55)
        try:
            confidence = min(0.85, max(0.30, float(raw_conf)))
        except Exception:
            confidence = 0.55

        candidate_text = f"{summary}\n" + "\n".join(principles)
        target = self._find_similar_meta(candidate_text, domain=domain)
        now = self._now()
        if target:
            target.summary = self._blend_text(target.summary, summary, limit=700)
            for p in principles:
                if p not in target.principles:
                    target.principles.append(p)
            target.principles = target.principles[:5]
            target.confidence = self._ema(target.confidence, confidence if reward > 0 else max(0.35, confidence - 0.1))
            target.evidence_count += 1
            target.reward_sum += reward
            target.updated_at = now
            target.source_path_ids = sorted({*target.source_path_ids, path_id})
            self.embeddings[target.id] = self._encode(self._meta_text(target))
            return target

        mid = f"m_{uuid.uuid4().hex[:10]}"
        node = MetaCognitionNode(
            id=mid,
            summary=summary,
            principles=principles,
            confidence=confidence,
            evidence_count=1,
            reward_sum=reward,
            domain=domain,
            source_path_ids=[path_id],
            created_at=now,
            updated_at=now,
        )
        self.meta[mid] = node
        self.embeddings[mid] = self._encode(self._meta_text(node))
        return node

    # ── weight updates ────────────────────────────────────────────────────────

    def _upsert_edge(self, source: str, target: str, relation: str, *, reward: float, base_weight: float) -> None:
        key = self._edge_key(source, target, relation)
        utility = base_weight if reward > 0 else max(0.15, base_weight * 0.65)
        edge = self.edges.get(key)
        if edge is None:
            self.edges[key] = GraphEdge(
                source=source,
                target=target,
                relation=relation,
                weight=utility,
                usage_count=1,
                reward_sum=reward,
            )
            return
        edge.weight = self._ema(edge.weight, utility)
        edge.usage_count += 1
        edge.reward_sum += reward

    def _edge(self, source: str, target: str, relation: str = "query_follows_path") -> Optional[GraphEdge]:
        return self.edges.get(self._edge_key(source, target, relation))

    @staticmethod
    def _edge_key(source: str, target: str, relation: str) -> str:
        return f"{source}|{relation}|{target}"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_similar_meta(self, text: str, *, domain: str) -> Optional[MetaCognitionNode]:
        if not self.meta:
            return None
        try:
            emb = self._encode(text)
        except Exception:
            return None
        best_score = -1.0
        best_node: Optional[MetaCognitionNode] = None
        for node in self.meta.values():
            if domain and node.domain and node.domain != domain:
                continue
            node_emb = self.embeddings.get(node.id)
            if node_emb is None:
                continue
            score = float(node_emb @ emb)
            if score > best_score:
                best_score = score
                best_node = node
        return best_node if best_node and best_score >= self.meta_merge_threshold else None

    def _trim(self) -> None:
        if len(self.meta) <= self.max_meta:
            return
        ranked = sorted(
            self.meta.values(),
            key=lambda m: (m.confidence, m.evidence_count, m.updated_at),
            reverse=True,
        )
        keep = ranked[: self.max_meta]
        keep_ids = {m.id for m in keep}
        self.meta = {m.id: m for m in keep}
        self.embeddings = {
            k: v
            for k, v in self.embeddings.items()
            if k in keep_ids or k.startswith("q_")
        }
        self.edges = {
            k: e
            for k, e in self.edges.items()
            if not e.target.startswith("m_") or e.target in keep_ids
        }

    def _encode(self, text: str) -> np.ndarray:
        emb = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb[0] if getattr(emb, "ndim", 1) == 2 else emb

    @staticmethod
    def _ema(old: float, new: float, alpha: float = 0.25) -> float:
        return (1.0 - alpha) * old + alpha * new

    @staticmethod
    def _blend_text(old: str, new: str, limit: int) -> str:
        if not new or new in old:
            return old[:limit]
        if old in new:
            return new[:limit]
        return f"{old} / {new}"[:limit]

    @staticmethod
    def _query_text(node: QueryNode) -> str:
        return (
            f"Domain: {node.domain}\nTask: {node.task}\nContext: {node.context}\n"
            f"Lesson: {node.lesson}\nOutcome: {node.outcome}"
        )

    @staticmethod
    def _meta_text(node: MetaCognitionNode) -> str:
        return f"{node.summary}\n" + "\n".join(node.principles)

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                print(f"    [TGM] LLM error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(3)
        return ""

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
