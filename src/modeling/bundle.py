"""Persistencia content-addressed de bundles de modelos.

Cada modelo se guarda como un directorio ``<version>`` con ``modelo.joblib`` y
``manifest.json``. La version es el SHA-256 de artefacto + manifest, de modo
que el mismo contenido produce la misma version y no se sobrescribe nada.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib

from .contracts import ModelAdapter


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_bundle(adapter: ModelAdapter, registry: Path,
                extra_metadata: dict | None = None) -> tuple[Path, dict]:
    """Guarda un adaptador y devuelve (ruta_artefacto, manifest congelado)."""
    registry.mkdir(parents=True, exist_ok=True)
    temporary = registry / f".modelo-{os.getpid()}.tmp"
    joblib.dump(adapter, temporary)
    artifact_hash = _sha256_bytes(temporary.read_bytes())

    manifest_payload = {
        "family": adapter.family,
        "name": adapter.name,
        "features": list(adapter.features),
        "artifact_sha256": artifact_hash,
        "artifact_file": "modelo.joblib",
        **(adapter.metadata() or {}),
        **(extra_metadata or {}),
    }
    manifest_bytes = json.dumps(
        manifest_payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    manifest_hash = _sha256_bytes(manifest_bytes)
    version_hash = _sha256_bytes((artifact_hash + manifest_hash).encode("ascii"))[:16]

    version_dir = registry / version_hash
    version_dir.mkdir(exist_ok=True)
    artifact_path = version_dir / "modelo.joblib"
    manifest_path = version_dir / "manifest.json"

    if artifact_path.exists() and _sha256_bytes(artifact_path.read_bytes()) != artifact_hash:
        raise ValueError(f"Colision de artefacto en {version_dir}")
    if not artifact_path.exists():
        temporary.replace(artifact_path)
    else:
        temporary.unlink()

    if manifest_path.exists() and _sha256_bytes(manifest_path.read_bytes()) != manifest_hash:
        raise ValueError(f"Manifest distinto en {version_dir}")
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_bytes)

    frozen = {**manifest_payload, "manifest_sha256": manifest_hash, "version": version_hash}
    return artifact_path, frozen


def load_bundle(registry: Path, version: str) -> tuple[ModelAdapter, dict]:
    """Carga un adaptador validando artefacto, manifest y version."""
    if not __import__("re").fullmatch(r"[0-9a-f]{16}", version or ""):
        raise ValueError("Version de bundle invalida")
    version_dir = registry / version
    artifact_path = version_dir / "modelo.joblib"
    manifest_path = version_dir / "manifest.json"
    if not artifact_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Bundle incompleto: {version_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = _sha256_bytes(manifest_path.read_bytes())
    artifact_hash = _sha256_bytes(artifact_path.read_bytes())
    if artifact_hash != manifest.get("artifact_sha256"):
        raise ValueError("artifact_sha256 no coincide")
    expected = _sha256_bytes((artifact_hash + manifest_hash).encode("ascii"))[:16]
    if version != expected:
        raise ValueError("version no corresponde a artefacto + manifest")
    return joblib.load(artifact_path), {**manifest, "manifest_sha256": manifest_hash,
                                        "version": version}


def publish_pointer(registry: Path, version: str, pointer: Path) -> None:
    """Actualiza el puntero activo de forma atomica."""
    manifest_path = registry / version / "manifest.json"
    manifest_hash = _sha256_bytes(manifest_path.read_bytes())
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(f".{pointer.name}-{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        {"version": version, "manifest_sha256": manifest_hash},
        indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(pointer)


def read_pointer(pointer: Path) -> dict:
    if not pointer.exists():
        raise FileNotFoundError(f"Falta puntero activo: {pointer}")
    return json.loads(pointer.read_text(encoding="utf-8"))