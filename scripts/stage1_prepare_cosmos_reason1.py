from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Tuple


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_json_array(path: str) -> Iterable[Dict[str, Any]]:
    """Iterate a JSON file that is a list of objects."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    for obj in data:
        if isinstance(obj, dict):
            yield obj


def _coerce_index2ans_to_choices(index2ans: Any) -> List[str]:
    if not isinstance(index2ans, dict):
        return []
    # keys are usually 'A','B','C','D'
    keys = sorted(index2ans.keys())
    out: List[str] = []
    for k in keys:
        v = index2ans[k]
        out.append(f"{str(k).upper()}. {str(v)}")
    return out


def _guess_fields(obj: Dict[str, Any]) -> Tuple[str, List[str], str, str]:
    """Return (prompt, choices, answer_key, id)."""

    # --- Cosmos-Reason1 Benchmark QA pairs format ---
    # {"video": "...", "qa_pairs": {"question": "...", "index2ans": {"A": "..."}, "answer": "C"}}
    if isinstance(obj.get("qa_pairs"), dict):
        qa = obj["qa_pairs"]
        prompt = qa.get("question") or ""
        choices = _coerce_index2ans_to_choices(qa.get("index2ans"))
        ans = str(qa.get("answer") or "").strip().upper()
        pid = obj.get("id") or obj.get("uid") or obj.get("question_id") or obj.get("qid")
        if pid is None:
            # include video for uniqueness
            video = str(obj.get("video") or "")
            pid = f"cosmos_{abs(hash(video + '|' + prompt))}"
        return prompt, choices, ans, str(pid)

    # --- generic schema fallback ---
    prompt = obj.get("prompt") or obj.get("question") or obj.get("input") or ""

    choices = obj.get("choices") or obj.get("options") or obj.get("candidates") or None
    if choices is None and isinstance(obj.get("A"), str):
        letters = [k for k in ["A", "B", "C", "D", "E"] if k in obj]
        if letters:
            choices = [f"{k}. {obj[k]}" for k in letters]

    if choices is None:
        choices = []
    else:
        out: List[str] = []
        for i, c in enumerate(list(choices)):
            if isinstance(c, dict) and "text" in c:
                c = c["text"]
            s = str(c)
            if not s[:2].upper().startswith(("A.", "B.", "C.", "D.", "E.")):
                letter = chr(ord("A") + i)
                s = f"{letter}. {s}"
            out.append(s)
        choices = out

    ans = obj.get("answer") or obj.get("label") or obj.get("answer_key") or obj.get("target") or ""
    ans = str(ans).strip()
    if len(ans) > 1:
        import re

        m = re.search(r"([A-E])", ans.upper())
        ans = m.group(1) if m else ans
    ans = ans.upper()

    pid = obj.get("id") or obj.get("uid") or obj.get("question_id") or obj.get("qid")
    if pid is None:
        pid = f"gen_{abs(hash(prompt))}"

    return prompt, choices, ans, str(pid)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/cosmos_reason1/raw")
    p.add_argument("--out", default="data/cosmos_reason1/prepared")
    p.add_argument("--max", type=int, default=0, help="max examples (0 means all)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Cosmos-Reason1 repo contains *.json files like */*_benchmark_qa_pairs.json
    json_paths = glob.glob(os.path.join(args.raw, "**/*.json"), recursive=True)
    jsonl_paths = glob.glob(os.path.join(args.raw, "**/*.jsonl"), recursive=True)

    candidates_json = [p for p in json_paths if p.endswith("_benchmark_qa_pairs.json")]
    candidates_jsonl = list(jsonl_paths)

    if not candidates_json and not candidates_jsonl:
        raise SystemExit(
            "No supported data file found under raw dir. Expected *_benchmark_qa_pairs.json or *.jsonl."
        )

    out_path = os.path.join(args.out, "all.jsonl")
    n = 0

    def write_obj(path: str, obj: Dict[str, Any]) -> None:
        nonlocal n
        prompt, choices, ans, pid = _guess_fields(obj)
        if not prompt or not choices or not ans:
            return

        meta = {"source_file": os.path.relpath(path, args.raw)}
        if isinstance(obj.get("video"), str):
            meta["video"] = obj.get("video")

        rec = {
            "id": pid,
            "prompt": prompt,
            "choices": choices,
            "answer_key": ans,
            "meta": meta,
        }
        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1

    with open(out_path, "w", encoding="utf-8") as w:
        # JSON array format first
        for path in sorted(candidates_json):
            for obj in _iter_json_array(path):
                write_obj(path, obj)
                if args.max and n >= args.max:
                    break
            if args.max and n >= args.max:
                break

        # JSONL fallback
        if not args.max or n < args.max:
            for path in sorted(candidates_jsonl):
                for obj in _iter_jsonl(path):
                    write_obj(path, obj)
                    if args.max and n >= args.max:
                        break
                if args.max and n >= args.max:
                    break

    print(out_path)
    print(f"prepared examples: {n}")


if __name__ == "__main__":
    main()
