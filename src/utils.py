# src/utils.py
import yaml
from copy import deepcopy

def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result

def load_config(override_path: str, default_path: str = "config/default.yaml") -> dict:
    with open(default_path) as f:
        base = yaml.safe_load(f) or {}
    with open(override_path) as f:
        override = yaml.safe_load(f) or {}
    return deep_merge(base, override)