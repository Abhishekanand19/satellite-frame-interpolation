# src/config/config_loader.py
"""
Central config loader. Single source of truth for all hyperparameters.
Switch dataset: goes → insat by changing one line in configs/dataset.yaml.
"""

import logging
from pathlib import Path
from typing import Any, Optional
import yaml

logger = logging.getLogger("config")

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


# ── Thin dot-access wrapper ───────────────────────────────────────────────────
class Config:
    """Recursive dot-access config object built from a dict."""

    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


# ── Loader ────────────────────────────────────────────────────────────────────
class ConfigLoader:

    def __init__(
        self,
        train_yaml:   str = str(CONFIGS / "train.yaml"),
        model_yaml:   str = str(CONFIGS / "model.yaml"),
        dataset_yaml: str = str(CONFIGS / "dataset.yaml"),
        overrides:    Optional[dict] = None,
    ):
        self._raw = {}
        for path in [train_yaml, model_yaml, dataset_yaml]:
            self._raw.update(self._load(path))

        if overrides:
            self._deep_update(self._raw, overrides)

        self._cfg = Config(self._raw)
        self._resolve_dataset()
        logger.info("Config loaded | dataset=%s", self.dataset_name)

    # ── Public interface ──────────────────────────────────────────────────────
    @property
    def train(self) -> Config:
        return self._cfg.training

    @property
    def model(self) -> Config:
        return self._cfg.model

    @property
    def dataset(self) -> Config:
        """Returns the active dataset sub-config (goes or insat) merged with globals."""
        return self._active_ds

    @property
    def dataset_name(self) -> str:
        return self._raw.get("dataset", "goes")

    @property
    def optimizer(self) -> Config:
        return self._cfg.optimizer

    @property
    def scheduler(self) -> Config:
        return self._cfg.scheduler

    @property
    def loss(self) -> Config:
        return self._cfg.loss

    @property
    def paths(self) -> Config:
        return self._cfg.paths

    @property
    def evaluation(self) -> Config:
        return self._cfg.evaluation

    @property
    def benchmark(self) -> Config:
        return self._cfg.benchmark

    @property
    def inference(self) -> Config:
        return self._cfg.model.inference

    def raw(self) -> dict:
        return self._raw

    # ── Internals ─────────────────────────────────────────────────────────────
    @staticmethod
    def _load(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        logger.debug("Loaded %s", path)
        return data

    @staticmethod
    def _deep_update(base: dict, override: dict) -> None:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                ConfigLoader._deep_update(base[k], v)
            else:
                base[k] = v

    def _resolve_dataset(self) -> None:
        """
        Merge global dataset keys with the active dataset sub-block.
        Result available as self.dataset — all other code reads only this.
        """
        name = self.dataset_name
        ds_raw = self._raw.get(name)
        if ds_raw is None:
            raise ValueError(
                f"Dataset '{name}' not defined in dataset.yaml. "
                f"Available: goes, insat"
            )
        merged = {
            "name":         name,
            "bt_min":       self._raw.get("bt_min",       180.0),
            "bt_max":       self._raw.get("bt_max",       320.0),
            "target_size":  self._raw.get("target_size",  [512, 512]),
            "stride":       self._raw.get("stride",       10),
            "max_triplets": self._raw.get("max_triplets", None),
            **ds_raw,
        }
        merged["target_size"] = tuple(merged["target_size"])
        self._active_ds = Config(merged)


# ── Convenience singleton loader ──────────────────────────────────────────────
_global: Optional[ConfigLoader] = None


def load_config(
    train_yaml:   Optional[str] = None,
    model_yaml:   Optional[str] = None,
    dataset_yaml: Optional[str] = None,
    overrides:    Optional[dict] = None,
) -> ConfigLoader:
    """
    Load (or return cached) global config.
    Pass overrides dict to patch values programmatically:
        load_config(overrides={"dataset": "insat"})
    """
    global _global
    _global = ConfigLoader(
        train_yaml   = train_yaml   or str(CONFIGS / "train.yaml"),
        model_yaml   = model_yaml   or str(CONFIGS / "model.yaml"),
        dataset_yaml = dataset_yaml or str(CONFIGS / "dataset.yaml"),
        overrides    = overrides,
    )
    return _global


def get_config() -> ConfigLoader:
    """Return already-loaded global config (must call load_config first)."""
    if _global is None:
        return load_config()
    return _global