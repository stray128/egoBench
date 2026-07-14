"""Config + path resolution.

Every path is resolved from the environment (DATA_ROOT / OUTPUT_ROOT) so the
same code and the same YAML run unchanged on the local box and on AWS. Nothing
downstream should ever hardcode an absolute path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()  # load .env if present; no-op otherwise
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    data_root: Path
    output_root: Path

    def dataset_dir(self, subdir: str) -> Path:
        return self.data_root / subdir

    def ensure(self) -> "Paths":
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        return self


def _resolve(env_key: str, default: str) -> Path:
    return Path(os.environ.get(env_key, default)).expanduser().resolve()


def paths() -> Paths:
    """Resolve DATA_ROOT / OUTPUT_ROOT from env, with repo-relative defaults."""
    return Paths(
        data_root=_resolve("DATA_ROOT", str(_REPO_ROOT / "data")),
        output_root=_resolve("OUTPUT_ROOT", str(_REPO_ROOT / "outputs")),
    )


def device() -> str:
    """Resolve compute device. 'auto' -> cuda if available, else cpu."""
    d = os.environ.get("DEVICE", "auto").lower()
    if d != "auto":
        return d
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the YAML config (default.yaml unless overridden)."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def disk_budget_gb() -> float:
    return float(os.environ.get("FETCH_DISK_BUDGET_GB", "25"))
