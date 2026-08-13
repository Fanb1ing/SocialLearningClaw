from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from socialclaw.memory import ContentAddressedArtifactStore, MemoryArtifactRef


class ContentAddressedArtifactStoreTests(unittest.TestCase):
    def test_grid_round_trip_is_lossless_and_deduplicated(self) -> None:
        grid = np.array([[0, 1], [15, 3]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            store = ContentAddressedArtifactStore(Path(directory) / "assets")
            pre = store.put_grid(grid, role="pre_state", metadata={"step": 0})
            post = store.put_grid(grid, role="post_state", metadata={"step": 1})
            loaded = store.load_grid(pre)
            files = list((Path(directory) / "assets" / "grids").glob("*.npy"))

        self.assertEqual(pre.relative_path, post.relative_path)
        self.assertEqual(pre.artifact_id, post.artifact_id)
        self.assertEqual(pre.role, "pre_state")
        self.assertEqual(post.role, "post_state")
        self.assertEqual(len(files), 1)
        self.assertEqual(loaded.dtype, np.dtype("<i2"))
        np.testing.assert_array_equal(loaded, grid)

    def test_png_is_verifiable_and_reuses_identical_bytes(self) -> None:
        image = Image.new("RGB", (4, 3), color=(10, 20, 30))
        with tempfile.TemporaryDirectory() as directory:
            store = ContentAddressedArtifactStore(Path(directory) / "assets")
            first = store.put_png(image, role="agent_view")
            second = store.put_png(image, role="keyframe")
            path = store.verify(first)

        self.assertEqual(first.relative_path, second.relative_path)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(path.suffix, ".png")
        self.assertEqual(first.metadata["width"], 4)
        self.assertEqual(first.metadata["height"], 3)

    def test_corruption_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ContentAddressedArtifactStore(Path(directory) / "assets")
            reference = store.put_grid(np.zeros((2, 2)), role="state")
            store.resolve(reference).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                store.verify(reference)

    def test_unsafe_reference_and_object_array_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MemoryArtifactRef(
                artifact_id="artifact_x",
                role="state",
                media_type="application/octet-stream",
                relative_path="../escape.bin",
                sha256="0" * 64,
                size_bytes=1,
            )
        with tempfile.TemporaryDirectory() as directory:
            store = ContentAddressedArtifactStore(directory)
            with self.assertRaisesRegex(ValueError, "Object arrays"):
                store.put_grid(
                    np.array([[object()]], dtype=object),
                    role="state",
                )


if __name__ == "__main__":
    unittest.main()
