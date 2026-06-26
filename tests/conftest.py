# tests/conftest.py
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def dummy_frame():
    """Single normalised BT frame (1,1,512,512)."""
    return torch.rand(1, 1, 512, 512, dtype=torch.float32)

@pytest.fixture(scope="session")
def dummy_pair(dummy_frame):
    t0 = torch.rand_like(dummy_frame)
    t2 = torch.rand_like(dummy_frame)
    return t0, t2

@pytest.fixture(scope="session")
def dummy_np():
    return (
        np.random.rand(512, 512).astype(np.float32),
        np.random.rand(512, 512).astype(np.float32),
    )

@pytest.fixture(scope="session")
def small_frame():
    """Tiny frame for fast tests."""
    return torch.rand(1, 1, 64, 64, dtype=torch.float32)