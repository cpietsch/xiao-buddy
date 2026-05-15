from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


@dataclass(frozen=True)
class ArtifactInstallResult:
    ok: bool
    detail: str
    path: Path | None = None
    installed: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_vector_data(
    *,
    target_data_path: Path,
    artifact_url: str,
    artifact_sha256: str = "",
    member_name: str = "",
    timeout: float = 60.0,
) -> ArtifactInstallResult:
    if target_data_path.exists():
        return ArtifactInstallResult(
            ok=True,
            detail=f"{target_data_path} already exists",
            path=target_data_path,
            installed=False,
        )
    if not artifact_url:
        return ArtifactInstallResult(
            ok=False,
            detail=f"{target_data_path} is missing and no artifact URL is configured",
        )
    return install_vector_artifact(
        artifact_url=artifact_url,
        target_data_path=target_data_path,
        artifact_sha256=artifact_sha256,
        member_name=member_name or target_data_path.name,
        timeout=timeout,
    )


def install_vector_artifact(
    *,
    artifact_url: str,
    target_data_path: Path,
    artifact_sha256: str = "",
    member_name: str = "",
    timeout: float = 60.0,
) -> ArtifactInstallResult:
    target_data_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path, cleanup_download = _materialize_artifact(artifact_url, timeout=timeout)
    try:
        expected_sha = _normalize_sha256(artifact_sha256)
        if expected_sha:
            actual_sha = sha256_file(downloaded_path)
            if actual_sha != expected_sha:
                return ArtifactInstallResult(
                    ok=False,
                    detail=f"artifact sha256 mismatch: expected {expected_sha}, got {actual_sha}",
                )

        tmp_data = target_data_path.with_name(f".{target_data_path.name}.tmp")
        if tmp_data.exists():
            tmp_data.unlink()

        try:
            if tarfile.is_tarfile(downloaded_path):
                _extract_tar_member(downloaded_path, tmp_data, member_name or target_data_path.name)
            else:
                shutil.copyfile(downloaded_path, tmp_data)
            tmp_data.replace(target_data_path)
        finally:
            if tmp_data.exists():
                tmp_data.unlink()

        return ArtifactInstallResult(
            ok=True,
            detail=f"installed vector artifact to {target_data_path}",
            path=target_data_path,
            installed=True,
        )
    finally:
        if cleanup_download and downloaded_path.exists():
            downloaded_path.unlink()


def _materialize_artifact(artifact_url: str, *, timeout: float) -> tuple[Path, bool]:
    parsed = urlparse(artifact_url)
    if parsed.scheme in {"http", "https"}:
        suffix = Path(parsed.path).suffix or ".artifact"
        handle = tempfile.NamedTemporaryFile(prefix="xiao-vector-", suffix=suffix, delete=False)
        temp_path = Path(handle.name)
        try:
            with handle:
                with requests.get(artifact_url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return temp_path, True

    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser(), False

    return Path(artifact_url).expanduser(), False


def _extract_tar_member(archive_path: Path, target_path: Path, member_name: str) -> None:
    with tarfile.open(archive_path) as archive:
        candidates = [
            member
            for member in archive.getmembers()
            if member.isfile() and Path(member.name).name == member_name
        ]
        if not candidates:
            raise FileNotFoundError(f"{archive_path} does not contain {member_name}")
        if len(candidates) > 1:
            raise ValueError(f"{archive_path} contains multiple files named {member_name}")
        member = candidates[0]
        source = archive.extractfile(member)
        if source is None:
            raise FileNotFoundError(f"could not read {member.name} from {archive_path}")
        with source, target_path.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def _normalize_sha256(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1]
    return value
