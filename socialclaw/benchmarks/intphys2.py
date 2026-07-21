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
        if split != "debug":
            raise ValueError("Only the locally prepared IntPhys2 debug split is supported")
        return self.data_dir / "debug_metadata.csv"

    def _videos_dir(self, split: str) -> Path:
        return self.data_dir / ("Debug/Videos" if split == "debug" else f"{split}/Videos")

    def load(self, split: str, max_samples: int = 0) -> List[BenchmarkSample]:
        import pandas as pd

        metadata_path = self._metadata_path(split)
        videos_dir = self._videos_dir(split)
        rows = pd.read_csv(metadata_path).to_dict("records")
        samples: List[BenchmarkSample] = []
        for row in rows:
            path = videos_dir / Path(str(row["file_name"])).name
            if not path.exists():
                continue
            label = 1 if "Possible" in str(row["type"]) else 0
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
                        "type": str(row.get("type", "")),
                        "game": str(row.get("game_name", "")),
                    },
                )
            )
            if max_samples and len(samples) >= max_samples:
                break
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
        return [str(self._metadata_path(split))]
