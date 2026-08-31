"""
Lightweight A-MEM-style agentic memory.

This implements the reusable pieces needed by the benchmark baselines:
structured memory notes, embedding retrieval, note linking, and a small
"memory evolution" pass that updates related notes after new experiences.
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
class MemoryNote:
    id: str
    content: str
    context: str = ""
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    n_updates: int = 0


class AMemory:
    """Agentic memory with local embeddings and LLM-assisted note evolution."""

    def __init__(
        self,
        client: OpenAI,
        model_id: str,
        embedder,
        max_notes: int = 80,
        retrieve_k: int = 5,
        evolve_k: int = 3,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.embedder = embedder
        self.max_notes = max_notes
        self.retrieve_k = retrieve_k
        self.evolve_k = evolve_k
        self.notes: List[MemoryNote] = []
        self.embeddings: Dict[str, np.ndarray] = {}

    def _encode(self, text: str) -> np.ndarray:
        emb = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb[0] if getattr(emb, "ndim", 1) == 2 else emb

    def _note_text(self, note: MemoryNote) -> str:
        return (
            f"{note.content}\n"
            f"Context: {note.context}\n"
            f"Keywords: {', '.join(note.keywords)}\n"
            f"Tags: {', '.join(note.tags)}"
        ).strip()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[MemoryNote]:
        if not self.notes:
            return []
        top_k = top_k or self.retrieve_k
        try:
            query_emb = self._encode(query)
        except Exception:
            return self.notes[-top_k:]

        scored = []
        for note in self.notes:
            emb = self.embeddings.get(note.id)
            if emb is None:
                continue
            score = float(emb @ query_emb)
            scored.append((score, note))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for score, n in scored[:top_k] if score > 0.05] or self.notes[-top_k:]

    def get_memory_block(self, query: str = "", top_k: Optional[int] = None) -> str:
        notes = self.retrieve(query, top_k=top_k) if query else self.notes[-(top_k or self.retrieve_k):]
        if not notes:
            return ""
        lines = []
        for i, n in enumerate(notes, 1):
            kws = f" Keywords: {', '.join(n.keywords[:6])}." if n.keywords else ""
            tags = f" Tags: {', '.join(n.tags[:4])}." if n.tags else ""
            lines.append(f"{i}. {n.content}{kws}{tags}")
        return (
            "=== Relevant agentic memory notes (apply when useful) ===\n"
            + "\n".join(lines)
            + "\n=========================================================\n"
        )

    def add_experience(
        self,
        *,
        task: str,
        outcome: bool,
        trajectory: str = "",
        lesson: str = "",
        context: str = "",
    ) -> MemoryNote:
        related = self.retrieve(f"{task}\n{trajectory}\n{lesson}", top_k=self.evolve_k)
        note = self._build_note(
            task=task,
            outcome=outcome,
            trajectory=trajectory,
            lesson=lesson,
            context=context,
            related=related,
        )
        now = datetime.utcnow().isoformat() + "Z"
        if not note.id:
            note.id = f"amem_{uuid.uuid4().hex[:10]}"
        note.created_at = note.created_at or now
        note.updated_at = note.updated_at or now
        note.links = sorted({*(note.links or []), *(r.id for r in related)})

        self.notes.append(note)
        self.embeddings[note.id] = self._encode(self._note_text(note))
        self._evolve_related(new_note=note, related=related)
        self._trim()
        return note

    def _build_note(
        self,
        *,
        task: str,
        outcome: bool,
        trajectory: str,
        lesson: str,
        context: str,
        related: List[MemoryNote],
    ) -> MemoryNote:
        related_text = "\n".join(f"- {n.content}" for n in related[: self.evolve_k]) or "(none)"
        prompt = (
            "Create one concise agent memory note from this experience.\n"
            "The note should be useful for solving future similar tasks.\n"
            "Return ONLY valid JSON with keys: content, context, keywords, tags.\n\n"
            f"Task:\n{task}\n\n"
            f"Outcome: {'success' if outcome else 'failure'}\n"
            f"Context:\n{context}\n\n"
            f"Trajectory or model output:\n{trajectory[:1200]}\n\n"
            f"Lesson:\n{lesson}\n\n"
            f"Related memory notes:\n{related_text}\n"
        )
        raw = self._call_llm(prompt, max_tokens=512)
        data = self._parse_json(raw)
        if not data:
            content = lesson or f"{'Successful' if outcome else 'Failed'} experience on: {task[:160]}"
            data = {
                "content": content,
                "context": context or task[:200],
                "keywords": self._keywords(task + " " + lesson),
                "tags": ["success" if outcome else "failure"],
            }
        return MemoryNote(
            id=f"amem_{uuid.uuid4().hex[:10]}",
            content=str(data.get("content", "")).strip()[:900],
            context=str(data.get("context", context or task[:240])).strip()[:600],
            keywords=[str(x).strip() for x in data.get("keywords", []) if str(x).strip()][:12],
            tags=[str(x).strip() for x in data.get("tags", []) if str(x).strip()][:8],
        )

    def _evolve_related(self, *, new_note: MemoryNote, related: List[MemoryNote]) -> None:
        if not related:
            return
        for old in related[: self.evolve_k]:
            prompt = (
                "A new memory note may refine an older memory note. "
                "If the older note should be updated, return JSON with an improved 'content'. "
                "If no change is needed, return JSON with the original content.\n\n"
                f"Older note:\n{old.content}\n\n"
                f"New note:\n{new_note.content}\n\n"
                "Return ONLY JSON: {\"content\": \"...\", \"keywords\": [...], \"tags\": [...]}."
            )
            raw = self._call_llm(prompt, max_tokens=384)
            data = self._parse_json(raw)
            if not data or not data.get("content"):
                continue
            new_content = str(data["content"]).strip()
            if new_content and new_content != old.content:
                old.content = new_content[:900]
                if isinstance(data.get("keywords"), list):
                    old.keywords = [str(x).strip() for x in data["keywords"] if str(x).strip()][:12]
                if isinstance(data.get("tags"), list):
                    old.tags = [str(x).strip() for x in data["tags"] if str(x).strip()][:8]
                old.updated_at = datetime.utcnow().isoformat() + "Z"
                old.n_updates += 1
                old.links = sorted({*old.links, new_note.id})
                self.embeddings[old.id] = self._encode(self._note_text(old))

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
                print(f"    [A-MEM] LLM error (attempt {attempt + 1}): {e}")
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
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

    @staticmethod
    def _keywords(text: str) -> List[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
        seen = []
        for w in words:
            if w not in seen:
                seen.append(w)
            if len(seen) >= 8:
                break
        return seen

    def _trim(self) -> None:
        if len(self.notes) <= self.max_notes:
            return
        keep = self.notes[-self.max_notes:]
        keep_ids = {n.id for n in keep}
        self.notes = keep
        self.embeddings = {k: v for k, v in self.embeddings.items() if k in keep_ids}

    def save(self, path: str | Path) -> None:
        payload = {"notes": [asdict(n) for n in self.notes]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.notes = [MemoryNote(**raw) for raw in data.get("notes", [])]
        self.embeddings = {}
        for n in self.notes:
            self.embeddings[n.id] = self._encode(self._note_text(n))
