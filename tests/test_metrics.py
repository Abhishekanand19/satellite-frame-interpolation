# tests/test_metrics.py
"""Tests for all physics and image quality metrics."""

import sys
import numpy as np
import torch
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physics_metrics.metrics import (
    mse, psnr, ssim, bt_mae, fsim, compute_all_metrics
)
from src.evaluation.evaluate import rmse, mae, compute_full_metrics


class TestMSE:
    def test_identical(self, dummy_np):
        p, _ = dummy_np
        assert mse(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        p = np.ones((4, 4), dtype=np.float32)
        g = np.zeros((4, 4), dtype=np.float32)
        assert mse(p, g) == pytest.approx(1.0)

    def test_symmetry(self, dummy_np):
        p, g = dummy_np
        assert mse(p, g) == pytest.approx(mse(g, p), rel=1e-5)

    def test_tensor_input(self, dummy_np):
        p, g = dummy_np
        tp = torch.from_numpy(p)
        tg = torch.from_numpy(g)
        assert mse(tp, tg) == pytest.approx(mse(p, g), rel=1e-5)


class TestPSNR:
    def test_identical_is_inf(self, dummy_np):
        p, _ = dummy_np
        assert psnr(p, p) == float("inf")

    def test_decreases_with_noise(self, dummy_np):
        p, _ = dummy_np
        low_noise  = np.clip(p + np.random.normal(0, 0.01, p.shape).astype(np.float32), 0, 1)
        high_noise = np.clip(p + np.random.normal(0, 0.1,  p.shape).astype(np.float32), 0, 1)
        assert psnr(low_noise, p) > psnr(high_noise, p)

    def test_range(self, dummy_np):
        p, g = dummy_np
        v = psnr(p, g)
        assert 0 < v < 100

    def test_data_range_param(self, dummy_np):
        p, g = dummy_np
        v1 = psnr(p, g, data_range=1.0)
        v2 = psnr(p * 255, g * 255, data_range=255.0)
        assert v1 == pytest.approx(v2, abs=0.1)


class TestSSIM:
    def test_identical_is_one(self, dummy_np):
        p, _ = dummy_np
        assert ssim(p, p) == pytest.approx(1.0, abs=1e-4)

    def test_range(self, dummy_np):
        p, g = dummy_np
        v = ssim(p, g)
        assert -1.0 <= v <= 1.0

    def test_higher_for_low_noise(self, dummy_np):
        p, _ = dummy_np
        low  = np.clip(p + np.random.normal(0, 0.01, p.shape).astype(np.float32), 0, 1)
        high = np.clip(p + np.random.normal(0, 0.2,  p.shape).astype(np.float32), 0, 1)
        assert ssim(low, p) > ssim(high, p)


class TestBTMAE:
    def test_identical(self, dummy_np):
        p, _ = dummy_np
        assert bt_mae(p, p) == pytest.approx(0.0, abs=1e-5)

    def test_unit_kelvin(self):
        p = np.ones((64, 64), dtype=np.float32)
        g = np.zeros((64, 64), dtype=np.float32)
        # diff of 1.0 in [0,1] → BT_MAX - BT_MIN = 140 K
        assert bt_mae(p, g) == pytest.approx(140.0, abs=0.1)

    def test_positive(self, dummy_np):
        p, g = dummy_np
        assert bt_mae(p, g) >= 0.0


class TestFSIM:
    def test_identical_near_one(self, dummy_np):
        p, _ = dummy_np
        v = fsim(p, p)
        assert v == pytest.approx(1.0, abs=0.01)

    def test_range(self, dummy_np):
        p, g = dummy_np
        v = fsim(p, g)
        assert 0.0 <= v <= 1.0


class TestComputeAllMetrics:
    def test_keys_present(self, dummy_np):
        p, g = dummy_np
        m = compute_all_metrics(p, g)
        for k in ["MSE", "PSNR", "SSIM", "BT_MAE", "FSIM"]:
            assert k in m

    def test_values_are_float(self, dummy_np):
        p, g = dummy_np
        m = compute_all_metrics(p, g)
        for v in m.values():
            assert isinstance(v, float)

    def test_perfect_prediction(self, dummy_np):
        p, _ = dummy_np
        m = compute_all_metrics(p, p)
        assert m["MSE"]  == pytest.approx(0.0,  abs=1e-7)
        assert m["SSIM"] == pytest.approx(1.0,  abs=1e-4)
        assert m["FSIM"] == pytest.approx(1.0,  abs=0.01)
        assert m["BT_MAE"] == pytest.approx(0.0, abs=1e-4)


class TestRMSEandMAE:
    def test_rmse_identical(self, dummy_np):
        p, _ = dummy_np
        assert rmse(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_mae_identical(self, dummy_np):
        p, _ = dummy_np
        assert mae(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_rmse_geq_mae(self, dummy_np):
        p, g = dummy_np
        assert rmse(p, g) >= mae(p, g) - 1e-6

    def test_compute_full_metrics_keys(self, dummy_np):
        p, g = dummy_np
        m = compute_full_metrics(p, g)
        for k in ["MSE", "PSNR", "SSIM", "BT_MAE", "FSIM", "RMSE", "MAE"]:
            assert k in m