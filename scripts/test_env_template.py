from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "xiao_copilot" / "config.py"
ENV_EXAMPLE_PATH = ROOT / ".env.example"


def main() -> None:
    expected = _settings_env_names(CONFIG_PATH)
    actual = _env_template_keys(ENV_EXAMPLE_PATH)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _assert(not missing, f".env.example is missing settings: {', '.join(missing)}")
    _assert(not extra, f".env.example has unknown settings: {', '.join(extra)}")

    content = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    forbidden_tokens = ("kitegg", "hs-mainz", "tailscale", "100.103.106.102")
    leaked = [token for token in forbidden_tokens if token.lower() in content.lower()]
    _assert(not leaked, f".env.example contains deployment-specific values: {', '.join(leaked)}")

    print("PASS env template regression")


def _settings_env_names(config_path: Path) -> set[str]:
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in {"_env_field", "_typed_env_field"}:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.isupper():
                names.add(arg.value)
                break
    return names


def _env_template_keys(path: Path) -> set[str]:
    _assert(path.exists(), ".env.example should exist")
    keys: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        _assert("=" in line, f".env.example line {line_number} is not KEY=VALUE")
        key, _, _value = line.partition("=")
        key = key.strip()
        _assert(key, f".env.example line {line_number} has an empty key")
        _assert(key not in keys, f".env.example repeats {key}")
        keys.add(key)
    return keys


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    sys.exit(main())
