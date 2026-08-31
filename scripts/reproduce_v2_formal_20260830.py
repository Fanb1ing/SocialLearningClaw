from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from socialclaw.dataset.arc_agi3 import arc_environment_fingerprint
from socialclaw.v2.efps import COGNITION_CONTRACT_VERSION
from socialclaw.v2.model import RecordedVisionModel
from socialclaw.v2.runtime import run_arc_online


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "v2_formal_20260830"
MANIFEST_PATH = EXPERIMENT_ROOT / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_files(experiment: dict[str, Any]) -> None:
    transcript = EXPERIMENT_ROOT / experiment["transcript"]
    actual_transcript = _sha256(transcript)
    if actual_transcript != experiment["transcript_sha256"]:
        raise RuntimeError(
            f"Frozen transcript hash mismatch for {experiment['name']}: "
            f"{actual_transcript}"
        )
    actual_environment = arc_environment_fingerprint(experiment["game_id"])
    if actual_environment != experiment["environment_fingerprint"]:
        raise RuntimeError(
            f"Vendored environment changed for {experiment['name']}: "
            f"{actual_environment}"
        )


def _verify_package_versions(required: dict[str, str]) -> None:
    mismatches = []
    for package, expected in required.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(
            "Reproduction package versions do not match:\n- "
            + "\n- ".join(mismatches)
            + "\nInstall experiments/v2_formal_20260830/requirements-reproduction.txt"
        )


def _verify_result_logs(
    output_dir: Path, experiment: dict[str, Any]
) -> dict[str, str]:
    actual: dict[str, str] = {}
    errors = []
    for relative_path, expected_hash in experiment["result_logs"].items():
        path = output_dir / relative_path
        if not path.is_file():
            errors.append(f"missing {relative_path}")
            continue
        actual_hash = _sha256(path)
        actual[relative_path] = actual_hash
        if actual_hash != expected_hash:
            errors.append(
                f"{relative_path}: expected {expected_hash}, got {actual_hash}"
            )
    if errors:
        raise RuntimeError(
            f"Result log verification failed for {experiment['name']}:\n- "
            + "\n- ".join(errors)
        )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-execute and byte-verify the three formal V2 runs from frozen "
            "logical model transcripts. No API key or network is used."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reproduced" / "v2_formal_20260830",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_contract = int(manifest["cognition_contract_version"])
    if expected_contract != COGNITION_CONTRACT_VERSION:
        raise SystemExit(
            "This frozen experiment uses cognition contract "
            f"{expected_contract}, but the current source uses contract "
            f"{COGNITION_CONTRACT_VERSION}. Refusing to present historical model "
            "responses as a result of the new Schema/Insight implementation."
        )
    _verify_package_versions(manifest["required_packages"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []

    for experiment in manifest["experiments"]:
        _verify_source_files(experiment)
        output_dir = output_root / experiment["output_dir_name"]
        if output_dir.exists():
            raise SystemExit(
                f"Refusing to overwrite existing reproduction directory: {output_dir}"
            )

        transcript_path = EXPERIMENT_ROOT / experiment["transcript"]
        model = RecordedVisionModel(transcript_path)
        summary = run_arc_online(
            output_dir,
            game_id=experiment["game_id"],
            model=model,
            max_steps=experiment["max_steps_per_level"],
            stop_after_levels=experiment["stop_after_levels"],
            reset_on_game_over=experiment["reset_on_game_over"],
            compact_process=experiment["compact_process"],
        )
        model.assert_exhausted()
        if summary != experiment["expected_summary"]:
            raise RuntimeError(
                f"Summary mismatch for {experiment['name']}:\n"
                + json.dumps(
                    {"expected": experiment["expected_summary"], "actual": summary},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        hashes = _verify_result_logs(output_dir, experiment)
        results.append(
            {
                "name": experiment["name"],
                "status": "passed",
                "output_dir": str(output_dir),
                "summary": summary,
                "verified_result_logs": hashes,
            }
        )
        print(
            f"[{experiment['name']}] passed: actions={summary['actions']}, "
            f"levels={summary['levels_passed']}/{summary['levels_attempted']}, "
            f"tokens={summary['usage']['total_tokens']}"
        )

    verification = {
        "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "status": "passed",
        "experiments": results,
    }
    verification_path = output_root / "reproduction_verification.json"
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"All result logs matched: {verification_path}")


if __name__ == "__main__":
    main()
