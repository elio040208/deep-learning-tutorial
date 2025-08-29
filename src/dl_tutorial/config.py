import argparse
import os
from typing import Any, Dict

import yaml


def load_config(config_path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_cli_overrides(kv_list: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for kv in kv_list:
        if "=" not in kv:
            continue
        key, raw_val = kv.split("=", 1)
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
        cursor[parts[-1]] = yaml.safe_load(raw_val)
    return overrides


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("-o", "--override", type=str, nargs="*", default=[], help="Override config as key=value, supports dots")
    return parser
