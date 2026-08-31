from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import List

from .base import BenchmarkAdapter, BenchmarkSample, Evaluation


def parse_binary_answer(text: str) -> int | None:
    stripped = (text or "").strip()
    if stripped in {"0", "1"}:
        return int(stripped)
    match = re.search(r"\b([01])\b", stripped)
    return int(match.group(1)) if match else None


class IntPhys2Benchmark(BenchmarkAdapter):
    name = "intphys2"
    default_split = "debug"

    def __init__(
        self,
        data_dir: str | Path = "data/intphys2",
        *,
        seconds_per_frame: float = 1.5,
        max_frames: int = 12,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seconds_per_frame = seconds_per_frame
        self.max_frames = max_frames

    def _metadata_path(self, split: str) -> Path:
        if split == "debug":
            official = self.data_dir / "Debug/metadata.csv"
            return official if official.exists() else self.data_dir / "debug_metadata.csv"
        if split == "main_300":
            return self.data_dir / "Main/sample_300.csv"
        raise ValueError("IntPhys2 split must be 'debug' or 'main_300'")

    def _videos_dir(self, split: str) -> Path:
        return self.data_dir / ("Debug/Videos" if split == "debug" else "Main/Videos")

    def _preparation_manifest_path(self, split: str) -> Path | None:
        if split == "main_300":
            return self.data_dir / "Main/sample_300_manifest.json"
        return None

    def load(self, split: str, max_samples: int = 0) -> List[BenchmarkSample]:
        import pandas as pd

        metadata_path = self._metadata_path(split)
        videos_dir = self._videos_dir(split)
        manifest_path = self._preparation_manifest_path(split)
        if manifest_path is not None and not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing pinned IntPhys2 preparation manifest: {manifest_path}. "
                "Run scripts/prepare_intphys2_data.py first."
            )
        rows = pd.read_csv(metadata_path).to_dict("records")
        samples: List[BenchmarkSample] = []
        missing_videos: List[Path] = []
        for row in rows:
            path = videos_dir / Path(str(row["file_name"])).name
            if not path.exists():
                missing_videos.append(path)
                continue
            sample_type = str(row["type"])
            if sample_type.endswith("_Possible"):
                label = 1
            elif sample_type.endswith("_Impossible"):
                label = 0
            else:
                raise ValueError(f"Unknown IntPhys2 type label: {sample_type!r}")
            if not max_samples or len(samples) < max_samples:
                samples.append(
                    BenchmarkSample(
                        id=path.stem,
                        prompt="Determine whether the video is physically plausible.",
                        gold=label,
                        media=[str(path)],
                        metadata={
                            "split": split,
                            "condition": str(row.get("condition", "unknown")).lower(),
                            "camera": str(row.get("Camera", "unknown")),
                            "type": sample_type,
                            "game": str(row.get("game_name", "")),
                        },
                    )
                )
        if missing_videos:
            preview = ", ".join(str(path) for path in missing_videos[:3])
            raise FileNotFoundError(
                f"IntPhys2 split {split!r} is incomplete: "
                f"{len(missing_videos)} referenced videos are missing; examples: {preview}"
            )
        return samples

    def extract_frame_data_urls(self, sample: BenchmarkSample) -> List[str]:
        import cv2

        if not sample.media:
            return []
        cap = cv2.VideoCapture(sample.media[0])
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        skip = max(1, int(fps * self.seconds_per_frame))
        frames: List[str] = []
        index = 0
        while len(frames) < self.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % skip == 0:
                encoded, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if encoded:
                    b64 = base64.b64encode(buffer).decode("ascii")
                    frames.append(f"data:image/jpeg;base64,{b64}")
            index += 1
        cap.release()
        return frames

    def evaluate(self, response: str, sample: BenchmarkSample) -> Evaluation:
        prediction = parse_binary_answer(response)
        return Evaluation(
            correct=prediction == int(sample.gold),
            prediction=prediction,
            details="" if prediction is not None else "invalid_binary_answer",
        )

    def system_prompt(self) -> str:
        return (
            "You evaluate whether a short simulated video obeys physical laws. Examine the "
            "ordered frames for object permanence, shape immutability, motion continuity, "
            "solidity, and gravity. Answer only 1 for physically plausible or 0 for impossible."
        )

    def rule_context(self) -> str:
        return (
            "Track each object across all frames. A violation requires positive evidence of "
            "disappearance, teleportation, impossible deformation, discontinuous motion, or "
            "interpenetration. Do not classify a scene as impossible merely because it is unusual."
        )

    def schema_query(self, sample: BenchmarkSample) -> str:
        """Include non-label scene metadata; never expose ``type`` or gold."""
        condition = sample.metadata.get("condition", "unknown")
        camera = sample.metadata.get("camera", "unknown")
        game = sample.metadata.get("game", "unknown")
        return (
            f"{sample.prompt}\nPhysical condition: {condition}. "
            f"Camera: {camera}. Scene family: {game}."
        )

    def source_files(self, split: str) -> List[str]:
        paths = [self._metadata_path(split)]
        manifest_path = self._preparation_manifest_path(split)
        if manifest_path is not None:
            paths.append(manifest_path)
        return [str(path) for path in paths]
