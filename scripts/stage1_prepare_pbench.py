from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional


def _normalize_choices(choices: Any) -> List[str]:
    if choices is None:
        return []
    out: List[str] = []
    if isinstance(choices, dict):
        for k in sorted(choices.keys()):
            out.append(f"{str(k).upper()}. {choices[k]}")
        return out
    if isinstance(choices, list):
        for i, c in enumerate(choices):
            if isinstance(c, dict) and "text" in c:
                c = c["text"]
            s = str(c)
            if not s[:2].upper().startswith(("A.", "B.", "C.", "D.", "E.")):
                s = f"{chr(ord('A') + i)}. {s}"
            out.append(s)
        return out
    return []


def _extract_answer_key(ans: Any) -> str:
    if ans is None:
        return ""
    s = str(ans).strip().upper()

    # common yes/no answers in PBench
    if s in {"YES", "Y", "TRUE", "T", "1"}:
        return "A"
    if s in {"NO", "N", "FALSE", "F", "0"}:
        return "B"

    if len(s) == 1 and "A" <= s <= "E":
        return s

    import re

    m = re.search(r"([A-E])", s)
    return m.group(1) if m else s


def _pick_first(obj: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _guess_record(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort parser for generic MCQ json/jsonl rows."""

    # prompt/question
    prompt = _pick_first(obj, ["prompt", "question", "query", "input", "text"])  # type: ignore[arg-type]

    # choices/options
    choices = _pick_first(obj, ["choices", "options", "candidates", "index2ans"])  # type: ignore[arg-type]

    # answer
    answer = _pick_first(obj, ["answer", "label", "answer_key", "target", "gt_answer"])  # type: ignore[arg-type]

    # id
    pid = _pick_first(obj, ["id", "uid", "qid", "question_id", "sample_id"])  # type: ignore[arg-type]

    prompt = str(prompt) if prompt is not None else ""
    choices_n = _normalize_choices(choices)
    ans_k = _extract_answer_key(answer)
    pid = str(pid) if pid is not None else f"gen_{abs(hash(prompt))}"

    if not prompt or not choices_n or not ans_k:
        return None

    meta: Dict[str, Any] = {}
    for k in ["image", "video", "media", "path", "scene", "split", "source", "task", "modality"]:
        if k in obj:
            meta[k] = obj[k]

    return {"id": pid, "prompt": prompt, "choices": choices_n, "answer_key": ans_k, "meta": meta}


def _iter_records_from_parquet(path: str) -> Iterable[Dict[str, Any]]:
    """Yield row dicts from a parquet file.

    Prefer pyarrow; fallback to pandas.
    """

    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        for row in table.to_pylist():
            if isinstance(row, dict):
                yield row
        return
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path)
        for row in df.to_dict(orient="records"):
            if isinstance(row, dict):
                yield row
        return
    except Exception as e:
        raise SystemExit(
            "Failed to read parquet. Please install pyarrow (recommended) or pandas. "
            f"Underlying error: {e}"
        )


def _pbench_rows_to_records(row: Dict[str, Any], *, source_file: str, images_out_dir: str) -> Iterable[Dict[str, Any]]:
    """Parse official nvidia/PBench parquet row into our prepared MCQ format.

    Parquet schema observed:
      - id: string
      - text_prompt: string
      - condition_image: struct<bytes: binary, path: string>
      - qa_pairs: string  # JSON string of list[{question, answer, category, subcategory}]

    We expand each qa pair into one binary (yes/no) MCQ problem.
    """

    if not isinstance(row, dict):
        return []

    base_id = str(row.get("id") or "")
    text_prompt = str(row.get("text_prompt") or "").strip()
    qa_pairs_raw = row.get("qa_pairs")

    if not base_id or not text_prompt or not qa_pairs_raw:
        return []

    try:
        qa_pairs = json.loads(qa_pairs_raw)
    except Exception:
        return []

    if not isinstance(qa_pairs, list):
        return []

    # extract / materialize conditioning image
    cond = row.get("condition_image") or {}
    cond_path = None
    if isinstance(cond, dict):
        cond_path = cond.get("path")

    # PBench snapshot often sets path=None and stores image bytes directly.
    img_bytes = cond.get("bytes") if isinstance(cond, dict) else None
    if (not cond_path) and isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 0:
        os.makedirs(images_out_dir, exist_ok=True)
        img_rel = os.path.join("images", f"{base_id}.jpg")
        img_abs = os.path.join(os.path.dirname(images_out_dir), img_rel)
        try:
            # avoid rewriting if exists
            if not os.path.exists(img_abs):
                with open(img_abs, "wb") as fw:
                    fw.write(bytes(img_bytes))
            cond_path = img_rel
        except Exception:
            # if write fails, silently skip attaching image
            cond_path = None

    out: List[Dict[str, Any]] = []
    for i, qa in enumerate(qa_pairs):
        if not isinstance(qa, dict):
            continue
        q = str(qa.get("question") or "").strip()
        a = qa.get("answer")
        if not q or a is None:
            continue

        # binary mcq
        choices = ["A. yes", "B. no"]
        ans_k = _extract_answer_key(a)
        if ans_k not in {"A", "B"}:
            # skip non-binary answers for now
            continue

        prompt = (
            "You are given a scene description and an image. Answer the question with the best option.\n\n"
            f"Scene description:\n{text_prompt}\n\n"
            f"Question: {q}"
        )

        meta: Dict[str, Any] = {
            "dataset": "pbench",
            "source_file": source_file,
            "pbench_id": base_id,
            "qa_index": i,
            "category": qa.get("category"),
            "subcategory": qa.get("subcategory"),
        }
        if cond_path:
            meta["image"] = cond_path

        out.append(
            {
                "id": f"{base_id}_{i}",
                "prompt": prompt,
                "choices": choices,
                "answer_key": ans_k,
                "meta": meta,
            }
        )

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/pbench/raw")
    p.add_argument("--out", default="data/pbench/prepared")
    p.add_argument("--max", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images_out_dir = os.path.join(args.out, "images")

    parquet_paths = glob.glob(os.path.join(args.raw, "**/*.parquet"), recursive=True)
    jsonl_paths = glob.glob(os.path.join(args.raw, "**/*.jsonl"), recursive=True)
    json_paths = glob.glob(os.path.join(args.raw, "**/*.json"), recursive=True)

    if not parquet_paths and not jsonl_paths and not json_paths:
        raise SystemExit(
            "No supported files found under raw. Expected .parquet/.jsonl/.json. Please inspect dataset structure."
        )

    out_path = os.path.join(args.out, "all.jsonl")
    n = 0

    with open(out_path, "w", encoding="utf-8") as w:
        # parquet first (PBench official release is parquet)
        for path in sorted(parquet_paths):
            rel = os.path.relpath(path, args.raw)
            for obj in _iter_records_from_parquet(path):
                # PBench path-specific parser
                if isinstance(obj, dict) and {"id", "text_prompt", "qa_pairs"}.issubset(obj.keys()):
                    for rec in _pbench_rows_to_records(obj, source_file=rel, images_out_dir=images_out_dir):
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                        if args.max and n >= args.max:
                            break
                    if args.max and n >= args.max:
                        break
                    continue

                # generic fallback
                rec = _guess_record(obj)
                if rec is None:
                    continue
                rec["meta"]["source_file"] = rel
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if args.max and n >= args.max:
                    break
            if args.max and n >= args.max:
                break

        # jsonl fallback
        if not args.max or n < args.max:
            for path in sorted(jsonl_paths):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            continue
                        rec = _guess_record(obj)
                        if rec is None:
                            continue
                        rec["meta"]["source_file"] = os.path.relpath(path, args.raw)
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                        if args.max and n >= args.max:
                            break
                if args.max and n >= args.max:
                    break

        # json fallback
        if not args.max or n < args.max:
            for path in sorted(json_paths):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                if isinstance(data, list):
                    for obj in data:
                        if not isinstance(obj, dict):
                            continue
                        rec = _guess_record(obj)
                        if rec is None:
                            continue
                        rec["meta"]["source_file"] = os.path.relpath(path, args.raw)
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                        if args.max and n >= args.max:
                            break
                if args.max and n >= args.max:
                    break

    print(out_path)
    print(f"prepared examples: {n}")


if __name__ == "__main__":
    main()
