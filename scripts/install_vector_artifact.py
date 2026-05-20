from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.vector_artifacts import install_vector_artifact
from xiao_copilot.vector_index import configured_index_paths


def main() -> None:
    args = _parse_args()
    manifest_path, fallback_data_path = configured_index_paths(args.manifest, args.data)
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = _resolve_data_path(meta, manifest_path, fallback_data_path, bool(args.data))
    result = install_vector_artifact(
        artifact_url=args.url,
        target_data_path=data_path,
        artifact_sha256=args.sha256,
        member_name=data_path.name,
        timeout=args.timeout,
    )
    print(result.detail)
    if not result.ok:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install a packaged vector artifact.")
    parser.add_argument("--url", required=True, help="HTTP(S), file://, or local path to artifact archive/data file.")
    parser.add_argument("--sha256", default="", help="Expected sha256 for the artifact archive or raw data file.")
    parser.add_argument("--manifest", default="", help="Vector manifest path.")
    parser.add_argument("--data", default="", help="Vector data path override.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Network timeout in seconds.")
    return parser.parse_args()


def _resolve_data_path(
    meta: dict[str, object],
    manifest_path: Path,
    fallback_data_path: Path,
    data_path_explicit: bool,
) -> Path:
    if data_path_explicit:
        return fallback_data_path
    data_file = meta.get("data_file")
    if data_file:
        path = Path(str(data_file))
        return path if path.is_absolute() else manifest_path.parent / path
    return fallback_data_path


if __name__ == "__main__":
    main()
