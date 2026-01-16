"""
Manifest management for artifact metadata and integrity.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from veritas_rag.core.contracts import ArtifactManifest


class ArtifactManifestManager:
    """Manages artifact manifest (metadata and checksums)."""

    @staticmethod
    def compute_file_checksum(file_path: Path) -> str:
        """
        Compute SHA256 checksum of a file.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def create_manifest(
        artifact_dir: Path,
        total_docs: int,
        total_chunks: int,
        index_type: str,
        compression: str,
        schema_version: str = "1.0",
        artifact_version: str = "1.0",
    ) -> ArtifactManifest:
        """
        Create manifest with file checksums.

        Args:
            artifact_dir: Directory containing artifact files
            total_docs: Total number of documents
            total_chunks: Total number of chunks
            index_type: Type of index ("bm25")
            compression: Compression method ("zstd")
            schema_version: Schema version
            artifact_version: Artifact version

        Returns:
            ArtifactManifest object
        """
        checksums = {}

        # Compute checksums for all artifact files
        files_to_check = ["chunks.bin", "chunks.idx", "bm25_index.pkl", "docs.meta"]

        for filename in files_to_check:
            file_path = artifact_dir / filename
            if file_path.exists():
                checksums[filename] = ArtifactManifestManager.compute_file_checksum(
                    file_path
                )

        return ArtifactManifest(
            schema_version=schema_version,
            artifact_version=artifact_version,
            build_timestamp=datetime.now(),
            total_docs=total_docs,
            total_chunks=total_chunks,
            index_type=index_type,
            compression=compression,
            checksums=checksums,
        )

    @staticmethod
    def save_manifest(manifest: ArtifactManifest, file_path: Path):
        """
        Save manifest to JSON file.

        Args:
            manifest: ArtifactManifest object
            file_path: Path to save manifest.json
        """
        manifest_dict = {
            "schema_version": manifest.schema_version,
            "artifact_version": manifest.artifact_version,
            "build_timestamp": manifest.build_timestamp.isoformat(),
            "total_docs": manifest.total_docs,
            "total_chunks": manifest.total_chunks,
            "index_type": manifest.index_type,
            "compression": manifest.compression,
            "checksums": manifest.checksums,
        }

        with open(file_path, "w") as f:
            json.dump(manifest_dict, f, indent=2)

    @staticmethod
    def load_manifest(file_path: Path) -> ArtifactManifest:
        """
        Load manifest from JSON file.

        Args:
            file_path: Path to manifest.json

        Returns:
            ArtifactManifest object
        """
        with open(file_path, "r") as f:
            manifest_dict = json.load(f)

        return ArtifactManifest(
            schema_version=manifest_dict["schema_version"],
            artifact_version=manifest_dict["artifact_version"],
            build_timestamp=datetime.fromisoformat(manifest_dict["build_timestamp"]),
            total_docs=manifest_dict["total_docs"],
            total_chunks=manifest_dict["total_chunks"],
            index_type=manifest_dict["index_type"],
            compression=manifest_dict["compression"],
            checksums=manifest_dict["checksums"],
        )

    @staticmethod
    def validate_manifest(manifest: ArtifactManifest, artifact_dir: Path) -> bool:
        """
        Validate manifest checksums (strict mode).

        Verifies:
        - Required files exist (chunks.bin, chunks.idx, bm25_index.pkl, docs.meta)
        - SHA256 checksums match for all files

        Args:
            manifest: ArtifactManifest object
            artifact_dir: Directory containing artifact files

        Returns:
            True if all checksums match, False otherwise
        """
        # Required files
        required_files = ["chunks.bin", "chunks.idx", "bm25_index.pkl", "docs.meta"]

        # Check all required files exist
        for filename in required_files:
            file_path = artifact_dir / filename
            if not file_path.exists():
                return False

        # Validate all checksums in manifest
        for filename, expected_checksum in manifest.checksums.items():
            file_path = artifact_dir / filename
            if not file_path.exists():
                return False

            actual_checksum = ArtifactManifestManager.compute_file_checksum(file_path)
            if actual_checksum != expected_checksum:
                return False

        return True
