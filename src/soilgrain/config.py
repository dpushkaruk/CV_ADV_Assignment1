from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(path=ROOT / "configs" / "base.yaml"):
    """configs/base.yaml as nested attributes: cfg.tiling.px_per_mm, cfg.data.dir, ..."""
    return _ns(yaml.safe_load(Path(path).read_text()))
