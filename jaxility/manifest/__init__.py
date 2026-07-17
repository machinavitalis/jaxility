# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Attestation manifest: schema, canonical serialisation, signing, verify.

Schema v0 is the OSS minimum (ADR-005): a BLAKE3 hash chain over
canonical JSON, binding the source attestation handle (from Jaxterity)
to the artifact content hash. The :class:`Signer` protocol is pluggable;
the OSS package ships :class:`HashChainSigner` (unsigned chain) and the
``jaxility-enterprise`` package plugs in real signing (ADR-008).

The canonical JSON serialiser :func:`canonical_dumps` lands in T-011
because every target hash (T-011) is computed over its output; the
Manifest schema itself lands in T-012.
"""

from .artifact import (
    ARTIFACT_SCHEMA_V0,
    Artifact,
    ArtifactCache,
    BuildLogEntry,
    default_cache_root,
)
from .canonical import canonical_dumps
from .models import SCHEMA_VERSION_V0, Manifest
from .signer import HASH_CHAIN_SIGNER_IDENTITY, HashChainSigner, Signer
from .toolchain_detect import (
    JAXILITY_ACADOS_DIR_ENV,
    JAXILITY_ACADOS_LIBRARY_PIN,
    JAXILITY_ACADOS_TEMPLATE_PIN,
    detect_acados_library_version,
    detect_acados_template_version,
    detect_casadi_version,
    detect_pinocchio_version,
    detect_toolchain_versions,
)
from .verify import (
    ChainReport,
    ManifestVerificationError,
    load_manifest,
    verify_cli,
    verify_manifest,
)

__all__ = [
    "ARTIFACT_SCHEMA_V0",
    "Artifact",
    "ArtifactCache",
    "BuildLogEntry",
    "ChainReport",
    "HASH_CHAIN_SIGNER_IDENTITY",
    "HashChainSigner",
    "JAXILITY_ACADOS_DIR_ENV",
    "JAXILITY_ACADOS_LIBRARY_PIN",
    "JAXILITY_ACADOS_TEMPLATE_PIN",
    "Manifest",
    "ManifestVerificationError",
    "SCHEMA_VERSION_V0",
    "Signer",
    "canonical_dumps",
    "default_cache_root",
    "detect_acados_library_version",
    "detect_acados_template_version",
    "detect_casadi_version",
    "detect_pinocchio_version",
    "detect_toolchain_versions",
    "load_manifest",
    "verify_cli",
    "verify_manifest",
]
