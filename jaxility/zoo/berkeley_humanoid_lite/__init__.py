# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Berkeley Humanoid Lite — CentroidalMPC on mock-cortex-a (stub source).

Berkeley Humanoid Lite ships in the Jaxility documentation but is
**not yet in the Jaxterity zoo**. The current entry
uses a :class:`~jaxility.testing.MockSource` stub so the Jaxility
pipeline can be exercised end-to-end now; Jaxterity will land a real
humanoid (floating-base + contact modelling) post-launch. See
:doc:`README.md`.
"""

from __future__ import annotations

from ...targets import MOCK_CORTEX_A
from ...testing import MockSource, Source
from .. import ZooDeploymentConfig

_BHL_HANDLE_SALT = b"jaxility-zoo-berkeley-humanoid-lite-stub"
"""Salt that pins the stub's attestation handle. Drop when the real
Jaxterity humanoid lands."""


def _source() -> Source:
    # 25-ish DoF stand-in (legs + arms + torso); the exact dimension is
    # cosmetic here — only the trajectory shape uses it.
    return MockSource(
        name="berkeley_humanoid_lite",
        initial_state=tuple(0.0 for _ in range(25)),
        dim=25,
        handle_salt=_BHL_HANDLE_SALT,
    )


def config() -> ZooDeploymentConfig:
    return ZooDeploymentConfig(
        name="berkeley_humanoid_lite",
        source_factory=_source,
        target=MOCK_CORTEX_A,
        template="CentroidalMPC",
        dtype="float64",
        n_steps=20,
        description=(
            "Berkeley Humanoid Lite, floating-base humanoid. Stub source; "
            "Jaxterity will land a real humanoid post-launch (rotor / "
            "contact modelling required, ADR-018 floating-base support "
            "already in Jaxterity)."
        ),
        license="MIT (Jaxility zoo entry).",
        upstream_status="stub-pending-jaxterity",
        remaining_work=(
            "Promote to real-robot when Jaxterity ships a humanoid zoo "
            "entry (floating-base + contact modelling).",
            "Land the centroidal MPC template (T-025).",
            "Real-target deployment lane (post-launch — T-080).",
        ),
    )
