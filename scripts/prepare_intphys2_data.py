#!/usr/bin/env python3
"""Prepare the complete IntPhys2 Debug set and a stratified Main subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "intphys2"
REPO_ID = "facebook/IntPhys2"
DEFAULT_REVISION = "a077a2f94e25889016fc6e5983cf21e2ddc25fb2"
STRATA = ("condition", "Camera", "Difficulty")
TYPE_ORDER = ("1_Possible", "1_Impossible", "2_Possible", "2_Impossible")
_THREAD_LOCAL = threading.local()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate_scenes(scene_rows: pd.DataFrame, count: int) -> dict[tuple, int]:
    groups = scene_rows.groupby(list(STRATA), sort=True, dropna=False).size()
    if count > len(scene_rows):
        raise ValueError(f"Requested {count} scenes from only {len(scene_rows)}")

    # Hamilton/largest-remainder allocation preserves the source distribution
    # while producing an exact integer scene count.
    raw = {key: count * int(size) / len(scene_rows) for key, size in groups.items()}
    allocation = {key: math.floor(value) for key, value in raw.items()}
    leftover = count - sum(allocation.values())
    ranked = sorted(
        groups.index,
        key=lambda key: (-(raw[key] - math.floor(raw[key])), tuple(map(str, key))),
    )
    for key in ranked:
        if not leftover:
            break
        if allocation[key] < int(groups[key]):
            allocation[key] += 1
            leftover -= 1
    if leftover:
        raise RuntimeError(f"Could not allocate {leftover} remaining scenes")
    return allocation


def _stratified_main_sample(
    metadata: pd.DataFrame,
    *,
    video_count: int,
    seed: int,
) -> pd.DataFrame:
    if video_count % len(TYPE_ORDER):
        raise ValueError(
            f"Main sample size must be divisible by {len(TYPE_ORDER)} "
            "so every selected scene keeps its four paired videos"
        )
    required = {"SceneIndex", "file_name", "type", *STRATA}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Main metadata is missing columns: {sorted(missing)}")

    per_scene = metadata.groupby("SceneIndex", sort=True)
    bad_types = [
        str(scene_id)
        for scene_id, rows in per_scene
        if set(rows["type"]) != set(TYPE_ORDER) or len(rows) != len(TYPE_ORDER)
    ]
    if bad_types:
        raise ValueError(
            "Expected exactly four paired type videos per Main scene; "
            f"invalid scenes include {bad_types[:5]}"
        )
    for column in STRATA:
        inconsistent = per_scene[column].nunique(dropna=False)
        if (inconsistent != 1).any():
            raise ValueError(f"{column} is inconsistent within a Main scene")

    scenes = metadata.drop_duplicates("SceneIndex").copy()
    scene_count = video_count // len(TYPE_ORDER)
    allocation = _allocate_scenes(scenes, scene_count)
    rng = np.random.default_rng(seed)
    selected_scene_ids: list[str] = []
    for key, rows in scenes.groupby(list(STRATA), sort=True, dropna=False):
        take = allocation[key]
        candidates = np.asarray(sorted(rows["SceneIndex"].tolist()))
        chosen = rng.choice(candidates, size=take, replace=False)
        selected_scene_ids.extend(str(value) for value in chosen)

    sample = metadata[metadata["SceneIndex"].isin(selected_scene_ids)].copy()
    sample["type"] = pd.Categorical(sample["type"], TYPE_ORDER, ordered=True)
    sample = sample.sort_values([*STRATA, "SceneIndex", "type"]).reset_index(drop=True)
    sample["type"] = sample["type"].astype(str)
    if len(sample) != video_count:
        raise RuntimeError(f"Expected {video_count} sampled videos, got {len(sample)}")
    return sample


def _download_files(
    filenames: list[str],
    *,
    revision: str,
    workers: int,
) -> None:
    failures: list[str] = []

    def session() -> requests.Session:
        cached = getattr(_THREAD_LOCAL, "session", None)
        if cached is None:
            cached = requests.Session()
            retries = Retry(
                total=5,
                connect=5,
                read=5,
                backoff_factor=1.0,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
            cached.mount("https://", HTTPAdapter(max_retries=retries))
            _THREAD_LOCAL.session = cached
        return cached

    def download(filename: str) -> str:
        destination = DATA_DIR / filename
        if destination.exists() and destination.stat().st_size > 0:
            return filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        encoded = quote(filename, safe="/")
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = (
            f"{endpoint}/datasets/{REPO_ID}/resolve/"
            f"{revision}/{encoded}?download=true"
        )
        with session().get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            expected = int(response.headers.get("Content-Length", "0"))
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if expected and temporary.stat().st_size != expected:
            raise IOError(
                f"size mismatch for {filename}: "
                f"{temporary.stat().st_size} != {expected}"
            )
        temporary.replace(destination)
        return filename

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, name): name for name in filenames}
        completed = 0
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{name}: {exc}")
            completed += 1
            if completed % 25 == 0 or completed == len(filenames):
                print(f"Downloaded/verified {completed}/{len(filenames)} files", flush=True)
    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(f"{len(failures)} downloads failed:\n{preview}")


def _distribution(rows: pd.DataFrame) -> dict:
    output = {}
    for column in (*STRATA, "type"):
        output[column] = {
            str(key): int(value)
            for key, value in rows[column].value_counts(dropna=False).sort_index().items()
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pinned Hugging Face dataset revision used for reproducible preparation",
    )
    args = parser.parse_args()

    if args.main_samples <= 0:
        raise SystemExit("--main-samples must be > 0")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")
    if not args.revision.strip():
        raise SystemExit("--revision must not be empty")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    revision = args.revision.strip()
    metadata_names = ["Debug/metadata.csv", "Main/metadata.csv"]
    _download_files(metadata_names, revision=revision, workers=min(args.workers, 2))

    debug_metadata_path = DATA_DIR / "Debug" / "metadata.csv"
    main_metadata_path = DATA_DIR / "Main" / "metadata.csv"
    debug = pd.read_csv(debug_metadata_path)
    main = pd.read_csv(main_metadata_path)
    sample = _stratified_main_sample(
        main,
        video_count=args.main_samples,
        seed=args.seed,
    )

    sample_path = DATA_DIR / "Main" / f"sample_{args.main_samples}.csv"
    sample.to_csv(sample_path, index=False)
    filenames = sorted(
        {f"Debug/{value}" for value in debug["file_name"]}
        | {f"Main/{value}" for value in sample["file_name"]}
    )
    _download_files(filenames, revision=revision, workers=args.workers)

    expected_debug = {
        Path(value).name for value in debug["file_name"]
    }
    expected_main = {
        Path(value).name for value in sample["file_name"]
    }
    actual_debug = {path.name for path in (DATA_DIR / "Debug" / "Videos").glob("*.mp4")}
    actual_main = {path.name for path in (DATA_DIR / "Main" / "Videos").glob("*.mp4")}
    missing_debug = sorted(expected_debug - actual_debug)
    missing_main = sorted(expected_main - actual_main)
    if missing_debug or missing_main:
        raise RuntimeError(
            f"Inventory verification failed: Debug missing={len(missing_debug)}, "
            f"Main missing={len(missing_main)}"
        )

    manifest = {
        "source": REPO_ID,
        "revision": revision,
        "seed": args.seed,
        "sampling_unit": "SceneIndex",
        "paired_types_per_scene": list(TYPE_ORDER),
        "strata": list(STRATA),
        "allocation": (
            "proportional Hamilton allocation across condition/camera/difficulty strata"
        ),
        "source_metadata_sha256": _sha256(main_metadata_path),
        "sample_csv": sample_path.name,
        "sample_csv_sha256": _sha256(sample_path),
        "debug_videos": len(expected_debug),
        "main_scenes": int(sample["SceneIndex"].nunique()),
        "main_videos": len(expected_main),
        "main_distribution": _distribution(sample),
        "debug_video_bytes": sum(
            (DATA_DIR / "Debug" / "Videos" / name).stat().st_size
            for name in expected_debug
        ),
        "main_video_bytes": sum(
            (DATA_DIR / "Main" / "Videos" / name).stat().st_size
            for name in expected_main
        ),
    }
    manifest_path = DATA_DIR / "Main" / f"sample_{args.main_samples}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
