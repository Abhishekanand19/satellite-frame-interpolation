# tests/test_dataset.py
"""Tests for data loading, preprocessing, and triplet construction."""

import sys
import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.goes_preprocessor import (
    normalize_bt, denormalize_bt, resize_array, to_tensor,
    BT_MIN, BT_MAX, TARGET_SIZE,
)
from src.data_loader.goes_dataset import build_triplets, GOESTripletDataset


# ── Preprocessing ─────────────────────────────────────────────────────────────
class TestNormalization:

    def test_output_range(self):
        bt = np.random.uniform(BT_MIN, BT_MAX, (512, 512)).astype(np.float32)
        out = normalize_bt(bt)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_dtype_float32(self):
        bt = np.ones((64, 64), dtype=np.float64) * 250.0
        assert normalize_bt(bt).dtype == np.float32

    def test_nan_handling(self):
        bt = np.full((64, 64), np.nan, dtype=np.float32)
        out = normalize_bt(bt)
        assert not np.isnan(out).any()

    def test_roundtrip(self):
        bt = np.random.uniform(BT_MIN, BT_MAX, (256, 256)).astype(np.float32)
        norm   = normalize_bt(bt)
        denorm = denormalize_bt(norm)
        np.testing.assert_allclose(bt, denorm, atol=1e-3)

    def test_clip_behaviour(self):
        bt = np.array([[100.0, 400.0]], dtype=np.float32)   # outside range
        out = normalize_bt(bt, clip=True)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_no_clip(self):
        bt = np.array([[100.0, 400.0]], dtype=np.float32)
        out = normalize_bt(bt, clip=False)
        assert out.min() < 0.0 or out.max() > 1.0


class TestResize:

    def test_output_shape(self):
        arr = np.random.rand(1024, 2048).astype(np.float32)
        out = resize_array(arr, (512, 512))
        assert out.shape == (512, 512)

    def test_dtype_preserved(self):
        arr = np.random.rand(128, 128).astype(np.float32)
        assert resize_array(arr, (64, 64)).dtype == np.float32

    def test_identity_size(self):
        arr = np.random.rand(512, 512).astype(np.float32)
        out = resize_array(arr, (512, 512))
        np.testing.assert_allclose(arr, out, atol=1e-4)


class TestToTensor:

    def test_shape(self, dummy_np):
        pred, _ = dummy_np
        t = to_tensor(pred)
        assert t.shape == (1, 1, 512, 512)

    def test_dtype(self, dummy_np):
        pred, _ = dummy_np
        assert to_tensor(pred).dtype == torch.float32

    def test_values_preserved(self, dummy_np):
        pred, _ = dummy_np
        t = to_tensor(pred)
        np.testing.assert_allclose(t.squeeze().numpy(), pred, atol=1e-6)


# ── Triplet construction ──────────────────────────────────────────────────────
class TestTriplets:

    def _fake_files(self, n=50):
        return [Path(f"frame_{i:04d}.nc") for i in range(n)]

    def test_triplet_count(self):
        files = self._fake_files(50)
        trips = build_triplets(files, stride=10)
        # expect n - 2*stride
        assert len(trips) == 50 - 2 * 10

    def test_triplet_ordering(self):
        files = self._fake_files(30)
        trips = build_triplets(files, stride=5)
        for t0, t1, t2 in trips:
            assert files.index(t0) < files.index(t1) < files.index(t2)

    def test_max_triplets_cap(self):
        files = self._fake_files(100)
        trips = build_triplets(files, stride=5, max_triplets=10)
        assert len(trips) == 10

    def test_insufficient_files(self):
        files = self._fake_files(5)
        trips = build_triplets(files, stride=10)
        assert len(trips) == 0


class TestGOESTripletDataset:

    def _mock_preprocess(self, *args, **kwargs):
        t = torch.rand(1, 1, 64, 64)
        return t, np.zeros((64, 64)), np.zeros((64, 64))

    def test_len(self):
        files = [Path(f"f_{i}.nc") for i in range(30)]
        trips = build_triplets(files, stride=5)
        ds    = GOESTripletDataset(trips, target_size=(64, 64))
        assert len(ds) == len(trips)

    def test_getitem_keys(self):
        files = [Path(f"f_{i}.nc") for i in range(30)]
        trips = build_triplets(files, stride=5)
        ds    = GOESTripletDataset(trips, target_size=(64, 64))

        with patch("src.data_loader.goes_dataset.preprocess_frame",
                   side_effect=self._mock_preprocess):
            sample = ds[0]

        assert "t0" in sample
        assert "t2" in sample
        assert "t1_gt" in sample
        assert sample["t0"].shape == (1, 64, 64)

    def test_getitem_tensor_range(self):
        files = [Path(f"f_{i}.nc") for i in range(30)]
        trips = build_triplets(files, stride=5)
        ds    = GOESTripletDataset(trips, target_size=(64, 64))

        with patch("src.data_loader.goes_dataset.preprocess_frame",
                   side_effect=self._mock_preprocess):
            sample = ds[0]

        for key in ["t0", "t2", "t1_gt"]:
            assert sample[key].min() >= 0.0
            assert sample[key].max() <= 1.0