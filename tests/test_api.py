# tests/test_api.py
"""Tests for FastAPI backend endpoints."""

import sys
import json
import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient


# ── Mock heavy dependencies before importing app ──────────────────────────────
@pytest.fixture(scope="module")
def client():
    mock_interp = MagicMock()
    mock_interp.model.return_value = (
        torch.rand(1, 1, 64, 64),
        torch.rand(1, 4, 64, 64),
        torch.rand(1, 1, 64, 64),
    )

    fake_tensor = torch.rand(1, 1, 64, 64)
    fake_raw    = np.random.rand(64, 64).astype(np.float32) * 140 + 180
    fake_norm   = np.random.rand(64, 64).astype(np.float32)

    with patch("src.interpolation.rife_model.RIFEThermalInterpolator",
               return_value=mock_interp), \
         patch("dashboard.backend.main.preprocess_frame",
               return_value=(fake_tensor, fake_raw, fake_norm)), \
         patch("dashboard.backend.main.discover_nc_files",
               return_value=[Path(f"frame_{i:04d}.nc") for i in range(50)]), \
         patch("dashboard.backend.main.build_triplets",
               return_value=[(Path("frame_0000.nc"),
                              Path("frame_0010.nc"),
                              Path("frame_0020.nc"))] * 10):

        from dashboard.backend.main import app
        yield TestClient(app)


class TestHealthEndpoint:
    def test_status_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_response_keys(self, client):
        r = client.get("/api/health")
        body = r.json()
        for k in ["status", "device", "cuda_available", "nc_files_found"]:
            assert k in body

    def test_status_value(self, client):
        r = client.get("/api/health")
        assert r.json()["status"] == "healthy"


class TestRootEndpoint:
    def test_root_ok(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_has_message(self, client):
        r = client.get("/")
        assert "message" in r.json()


class TestFilesEndpoint:
    def test_status(self, client):
        r = client.get("/api/files")
        assert r.status_code == 200

    def test_has_total(self, client):
        r = client.get("/api/files")
        assert "total" in r.json()

    def test_total_is_int(self, client):
        r = client.get("/api/files")
        assert isinstance(r.json()["total"], int)

    def test_files_list_present(self, client):
        r = client.get("/api/files")
        assert "files" in r.json()


class TestTripletsEndpoint:
    def test_status(self, client):
        r = client.get("/api/triplets")
        assert r.status_code == 200

    def test_structure(self, client):
        r = client.get("/api/triplets")
        body = r.json()
        assert "total" in body
        assert "triplets" in body

    def test_stride_param(self, client):
        r = client.get("/api/triplets?stride=10&limit=5")
        assert r.status_code == 200
        assert r.json()["stride_minutes"] == 10


class TestInterpolateEndpoint:
    def test_valid_request(self, client):
        r = client.post("/api/interpolate",
                        json={"triplet_index": 0, "stride": 10})
        assert r.status_code == 200

    def test_response_has_images(self, client):
        r = client.post("/api/interpolate",
                        json={"triplet_index": 0, "stride": 10})
        body = r.json()
        assert "images" in body
        for k in ["t0", "t2", "prediction", "diff_heatmap", "confidence"]:
            assert k in body["images"]

    def test_response_has_metrics(self, client):
        r = client.post("/api/interpolate",
                        json={"triplet_index": 0, "stride": 10})
        assert "metrics" in r.json()

    def test_response_has_bt_stats(self, client):
        r = client.post("/api/interpolate",
                        json={"triplet_index": 0, "stride": 10})
        assert "bt_stats" in r.json()

    def test_images_are_base64_strings(self, client):
        r = client.post("/api/interpolate",
                        json={"triplet_index": 0, "stride": 10})
        images = r.json()["images"]
        for k, v in images.items():
            assert isinstance(v, str) and len(v) > 100, f"{k} not valid b64"

    def test_inference_time_present(self, client):
        r = client.post("/api/interpolate",
                        json={"triplet_index": 0, "stride": 10})
        assert "inference_time_s" in r.json()

    def test_out_of_range_triplet(self, client):
        r = client.post("/api/interpolate",
                        json={"triplet_index": 99999, "stride": 10})
        assert r.status_code == 400

    def test_invalid_payload(self, client):
        r = client.post("/api/interpolate", json={})
        assert r.status_code == 422


class TestFrameEndpoint:
    def test_valid_index(self, client):
        r = client.get("/api/frame/0")
        assert r.status_code == 200

    def test_response_keys(self, client):
        r = client.get("/api/frame/0")
        body = r.json()
        for k in ["index", "file", "image_b64", "stats"]:
            assert k in body

    def test_stats_keys(self, client):
        r = client.get("/api/frame/0")
        stats = r.json()["stats"]
        for k in ["bt_min", "bt_max", "bt_mean"]:
            assert k in stats

    def test_out_of_range(self, client):
        r = client.get("/api/frame/99999")
        assert r.status_code == 404