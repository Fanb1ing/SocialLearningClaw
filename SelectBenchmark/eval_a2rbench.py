#!/usr/bin/env python3
"""
Evaluate A2RBench benchmark.

A2RBench tests abstract reasoning: given a rule description + input→output examples,
apply the same rule to a new input. Evaluation metric: Exact Match Accuracy (%).

Paper: https://arxiv.org/abs/2605.17278
Dataset: github.com/MAC-AutoML/A2Rbench (questions_arc_text_v9/)

Task types:
  - SymbolicRule: string transformation based on positional rules (1D/2D/3D)
  - SemanticRule: character-level mapping based on semantic rules (1D/2D/3D)

Models:
  - claude-opus-4.8: current strongest Claude (NOT in original paper)
  - gemini-2.5-flash: paper baseline (listed in paper's answer.py as Answerer_Gemini_Flash)

Paper reports top models around 39.8% accuracy on representative subset; humans ~68.5%.

Usage:
    cd /data5/fanbingbing/SocialLearningClaw
    .venv/bin/python SelectBenchmark/eval_a2rbench.py
"""

import os, re, json, time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODELS = {
    "claude-opus-4.8":  "anthropic/claude-opus-4.8",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
}

# Paper reports overall accuracy ~39.8% for top models (human: 68.5%)
# No per-model breakdown available in public readme; confirmed from paper text
PAPER_INFO = {
    "gemini-2.5-flash": "in paper (Answerer_Gemini_Flash)",
    "human": 68.5,
    "top_model": 39.8,
}

DATA_FILE = Path(__file__).parent / "data" / "A2RBench" / "validated_questions.jsonl"
OUT_DIR   = Path(__file__).parent / "results" / "a2rbench"
MAX_SAMPLES = 20  # out of 72 total; balanced across task_type and dimensionality

SYSTEM_PROMPT = (
    "You are a powerful and logical AI specializing in abstract reasoning. "
    "Your goal is to deduce the hidden transformation rule from examples and apply it. "
    "Output MUST be a single raw JSON object with fields 'reasoning' and 'final_answer'."
)

USER_TEMPLATE = """Analyze the examples below to deduce the transformation rule, then apply it to the question.

**Rule Description:**
{rule_description}

**Examples (input → output):**
{examples_str}

**Question:** Apply the same rule to: {question}

Output exactly this JSON format (no markdown, no extra text):
{{"reasoning": "step-by-step reasoning here", "final_answer": <your answer matching the output type>}}"""


def format_examples(examples: list[dict], max_ex: int = 5) -> str:
    lines = []
    for i, ex in enumerate(examples[:max_ex]):
        inp = json.dumps(ex["input"]) if not isinstance(ex["input"], str) else repr(ex["input"])
        out = json.dumps(ex["output"]) if not isinstance(ex["output"], str) else repr(ex["output"])
        lines.append(f"  - {inp} → {out}")
    return "\n".join(lines)


def call_model(model_id: str, prompt_user: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt_user},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return ""


def extract_answer(text: str):
    """Parse final_answer from model JSON output."""
    # Try direct JSON parse
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        obj = json.loads(cleaned)
        return obj.get("final_answer")
    except json.JSONDecodeError:
        pass
    # Fallback: regex extract final_answer value
    m = re.search(r'"final_answer"\s*:\s*(".*?"|null|\[.*?\]|\{.*?\})', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return m.group(1).strip('"')
    return None


def normalize(val) -> str:
    """Normalize answer for comparison."""
    if val is None:
        return ""
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    return str(val).strip()


def answers_match(pred, gold) -> bool:
    return normalize(pred) == normalize(gold)


def load_samples(max_n: int = MAX_SAMPLES) -> list[dict]:
    """Load balanced sample: equal task_type × dimensionality distribution."""
    from collections import defaultdict
    data = [json.loads(l) for l in open(DATA_FILE)]
    buckets = defaultdict(list)
    for d in data:
        key = (d["task_type"], d["dimensionality"])
        buckets[key].append(d)
    # Take floor(max_n / n_buckets) from each bucket
    n_buckets = len(buckets)
    per_bucket = max(1, max_n // n_buckets)
    samples = []
    for key, items in sorted(buckets.items()):
        samples.extend(items[:per_bucket])
    return samples[:max_n]


def evaluate(model_name: str, model_id: str, samples: list[dict]) -> dict:
    records = []
    n_correct = 0
    print(f"\n  [{model_name}] {len(samples)} samples")

    for i, item in enumerate(samples):
        pd = item["puzzle_data"]
        question = pd["question_plaintext"]
        gold     = pd["answer_ciphertext"]
        examples = pd.get("examples", [])

        prompt = USER_TEMPLATE.format(
            rule_description=item["rule_description"][:1500],
            examples_str=format_examples(examples),
            question=json.dumps(question) if not isinstance(question, str) else repr(question),
        )

        output = call_model(model_id, prompt)
        pred   = extract_answer(output)
        ok     = answers_match(pred, gold)
        n_correct += ok

        records.append({
            "id":         item["question_id"],
            "task_type":  item["task_type"],
            "dim":        item["dimensionality"],
            "gold":       normalize(gold),
            "predicted":  normalize(pred),
            "correct":    ok,
            "output_snippet": output[:400],
        })
        print(f"    [{i+1:02d}/{len(samples)}] {'✓' if ok else '✗'}  "
              f"type={item['task_type'][:3]}  dim={item['dimensionality']}  "
              f"pred={normalize(pred)[:30]!r}  gold={normalize(gold)[:30]!r}")
        time.sleep(1)

    acc = 100.0 * n_correct / len(samples) if samples else 0.0
    print(f"  => Accuracy: {acc:.1f}%  ({n_correct}/{len(samples)})")
    return {"accuracy": acc, "n_correct": n_correct, "n_total": len(samples), "records": records}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    print(f"Loaded {len(samples)} samples from {DATA_FILE}")
    from collections import Counter
    dist = Counter((s["task_type"], s["dimensionality"]) for s in samples)
    for k, v in sorted(dist.items()):
        print(f"  {k[0][:3]}-D{k[1]}: {v}")

    all_results = {}
    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}\nModel: {model_name}  ({model_id})\n{'='*60}")
        result = evaluate(model_name, model_id, samples)
        all_results[model_name] = result

        # Per type/dim breakdown
        from collections import defaultdict
        by_type = defaultdict(lambda: {"c": 0, "n": 0})
        for r in result["records"]:
            k = f"{r['task_type'][:3]}-D{r['dim']}"
            by_type[k]["n"] += 1
            by_type[k]["c"] += r["correct"]
        print("  Breakdown:")
        for k, v in sorted(by_type.items()):
            print(f"    {k}: {100.*v['c']/v['n']:.0f}% (n={v['n']})")
        result["by_category"] = {k: {"accuracy": 100.*v["c"]/v["n"], "n": v["n"]}
                                  for k, v in by_type.items()}

        with open(OUT_DIR / f"{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n\n" + "="*65)
    print("RESULTS vs PAPER  (metric: Exact Match Accuracy %)")
    print("  Paper: top models ~39.8%, humans ~68.5%")
    print("="*65)
    print(f"{'Model':<25} {'Accuracy':>10} {'Note':>20}")
    print("-"*55)
    for m, r in all_results.items():
        note = "NOT in paper" if m == "claude-opus-4.8" else PAPER_INFO.get(m, "")
        print(f"{m:<25} {r['accuracy']:>9.1f}% {note:>20}")
    print(f"{'Human (paper)':<25} {'~68.5%':>10}")
    print(f"{'Top model (paper)':<25} {'~39.8%':>10}")

    summary = {
        "metric": "Exact Match Accuracy (%) — match predicted final_answer to answer_ciphertext",
        "our_results": {m: r["accuracy"] for m, r in all_results.items()},
        "paper_info":  PAPER_INFO,
        "n_samples":   len(samples),
        "note": "72 total validated problems; sampled 20 balanced across task_type × dim",
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
