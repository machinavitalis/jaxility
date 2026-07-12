# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Jaxterity attestation-chain integration tests (T-016).

End-to-end: load a real Jaxterity zoo robot, run it through the mock
lowering pipeline, confirm the attestation chain from the robot's
handle through the manifest to the artifact is intact, then mutate
each of the four canonical inputs Jaxterity's handle reacts to (URDF,
calibration parameters, telemetry hash, recipe version) and confirm
the artifact hash changes.

If Jaxterity isn't importable the suite skips cleanly; in the bootstrap
environment Jaxterity is installed editable from ~/Dev/jaxterity
so these tests run on every local preflight.
"""

from __future__ import annotations

import pytest

jaxterity = pytest.importorskip("jaxterity")
from jaxterity.zoo import load  # noqa: E402

from jaxility.errors import SourceError  # noqa: E402
from jaxility.manifest import verify_manifest  # noqa: E402
from jaxility.targets import MOCK_CORTEX_A  # noqa: E402
from jaxility.testing import (  # noqa: E402
    JaxteritySource,
    Source,
    compare,
    mock_lower,
)


def _cartpole_source() -> JaxteritySource:
    """Load Jaxterity's Cartpole zoo robot and wrap it as a Source."""
    robot = load("cartpole")
    # Zoo robots are UNCALIBRATED; the deployment compiler will
    # require CALIBRATED in production. We opt in by passing None.
    return JaxteritySource.from_robot(robot, dim=2, require_calibration_state=None)


# ---------------------------------------------------------------------------
# Acceptance 1: chain is intact end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_jaxterity_source_satisfies_protocol() -> None:
    """The adapter registers as a Jaxility ``Source`` at runtime."""
    assert isinstance(_cartpole_source(), Source)


@pytest.mark.unit
def test_artifact_manifest_source_handle_matches_robot_handle() -> None:
    """The chain's anchor is the robot's BLAKE3 handle (invariant 2)."""
    robot = load("cartpole")
    source = JaxteritySource.from_robot(robot, require_calibration_state=None)
    bundle = mock_lower(source, MOCK_CORTEX_A)

    # Jaxterity exposes the handle as a hex string; Jaxility stores bytes.
    expected = bytes.fromhex(robot.attestation_handle)
    assert bundle.manifest.source_attestation_handle == expected
    assert bundle.manifest.source_attestation_handle == source.attestation_handle


@pytest.mark.unit
def test_full_chain_links_unbroken() -> None:
    """Every chain hop verifies: robot → source → manifest → artifact."""
    robot = load("cartpole")
    source = JaxteritySource.from_robot(robot, require_calibration_state=None)
    bundle = mock_lower(source, MOCK_CORTEX_A)

    # robot handle == source handle (hex → bytes).
    assert source.attestation_handle == bytes.fromhex(robot.attestation_handle)
    # source handle == manifest source_attestation_handle.
    assert bundle.manifest.source_attestation_handle == source.attestation_handle
    # manifest artifact_content_hash == artifact content_hash.
    assert bundle.manifest.artifact_content_hash == bundle.artifact.content_hash
    # artifact source_manifest_hash == manifest content_hash().
    assert bundle.artifact.source_manifest_hash == bundle.manifest.content_hash()
    # target hash agrees on both ends.
    assert bundle.manifest.target_profile_hash == MOCK_CORTEX_A.hash
    assert bundle.artifact.target_profile_hash == MOCK_CORTEX_A.hash


@pytest.mark.unit
def test_manifest_verifies_under_oss_signer() -> None:
    """The full chain verifies under :class:`HashChainSigner`."""
    bundle = mock_lower(_cartpole_source(), MOCK_CORTEX_A)
    report = verify_manifest(bundle.manifest)
    assert report.ok is True


@pytest.mark.unit
def test_equivalence_passes_on_jaxterity_source() -> None:
    """Mock-lowered Cartpole's bundle simulate matches source simulate."""
    source = _cartpole_source()
    bundle = mock_lower(source, MOCK_CORTEX_A)
    report = compare(
        source.simulate(50),
        bundle.simulate(50),
        target_family="mock-cortex-a",
        dtype="float64",
    )
    assert report.overall_passed is True


# ---------------------------------------------------------------------------
# Acceptance 2: mutating each of the four canonical inputs changes the chain.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_urdf_mutation_changes_artifact_hash() -> None:
    """A different model (cartpole vs so100) → different artifact hash.

    Jaxterity does not expose a Robot.with_urdf mutator (the URDF
    arrives once via the zoo loader); we model "URDF changed" as
    "loaded a different model". Both cartpole and so100 ship in
    jaxterity.zoo.
    """
    cartpole = JaxteritySource.from_robot(
        load("cartpole"), require_calibration_state=None
    )
    so100 = JaxteritySource.from_robot(
        load("so100"), require_calibration_state=None, dim=6
    )

    assert cartpole.attestation_handle != so100.attestation_handle

    bundle_cp = mock_lower(cartpole, MOCK_CORTEX_A)
    bundle_so = mock_lower(so100, MOCK_CORTEX_A)
    assert bundle_cp.artifact.content_hash != bundle_so.artifact.content_hash


@pytest.mark.unit
def test_calibration_parameter_mutation_changes_artifact_hash() -> None:
    """Robot.with_parameters({...}) flips the handle → flips the chain."""
    base = load("cartpole")
    perturbed = base.with_parameters(
        {"cart.mass": base.parameters()["cart.mass"] * 2.0}
    )

    assert base.attestation_handle != perturbed.attestation_handle

    src_base = JaxteritySource.from_robot(base, require_calibration_state=None)
    src_perturbed = JaxteritySource.from_robot(
        perturbed, require_calibration_state=None
    )

    bundle_base = mock_lower(src_base, MOCK_CORTEX_A)
    bundle_perturbed = mock_lower(src_perturbed, MOCK_CORTEX_A)
    assert bundle_base.artifact.content_hash != bundle_perturbed.artifact.content_hash


@pytest.mark.unit
def test_telemetry_hash_mutation_changes_artifact_hash() -> None:
    """Different telemetry-hash provenance → different handle → different chain."""
    base = load("cartpole")
    calibrated_a = base.with_provenance(
        ("phase1-recipe", "v0.0.1", "tel-hash-aaaaaa"),
        calibrated=True,
    )
    calibrated_b = base.with_provenance(
        ("phase1-recipe", "v0.0.1", "tel-hash-bbbbbb"),
        calibrated=True,
    )
    assert calibrated_a.attestation_handle != calibrated_b.attestation_handle

    bundle_a = mock_lower(
        JaxteritySource.from_robot(
            calibrated_a, require_calibration_state="CALIBRATED"
        ),
        MOCK_CORTEX_A,
    )
    bundle_b = mock_lower(
        JaxteritySource.from_robot(
            calibrated_b, require_calibration_state="CALIBRATED"
        ),
        MOCK_CORTEX_A,
    )
    assert bundle_a.artifact.content_hash != bundle_b.artifact.content_hash


@pytest.mark.unit
def test_recipe_version_mutation_changes_artifact_hash() -> None:
    """Different recipe version → different handle → different chain."""
    base = load("cartpole")
    v0 = base.with_provenance(
        ("phase1-recipe", "v0.0.1", "tel-hash-shared"), calibrated=True
    )
    v1 = base.with_provenance(
        ("phase1-recipe", "v0.0.2", "tel-hash-shared"), calibrated=True
    )
    assert v0.attestation_handle != v1.attestation_handle

    bundle_v0 = mock_lower(
        JaxteritySource.from_robot(v0, require_calibration_state="CALIBRATED"),
        MOCK_CORTEX_A,
    )
    bundle_v1 = mock_lower(
        JaxteritySource.from_robot(v1, require_calibration_state="CALIBRATED"),
        MOCK_CORTEX_A,
    )
    assert bundle_v0.artifact.content_hash != bundle_v1.artifact.content_hash


# ---------------------------------------------------------------------------
# Chain "broken on purpose" — verifies clearly that the broken case is broken.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_broken_chain_is_detected() -> None:
    """Replacing ``source_attestation_handle`` flips the recomputed hash."""
    source = _cartpole_source()
    bundle = mock_lower(source, MOCK_CORTEX_A)

    # Round-trip the manifest with the source handle replaced.
    payload = bundle.manifest.model_dump(mode="json")
    payload["source_attestation_handle"] = "00" * 32  # zeros — not the real handle
    from jaxility.manifest import Manifest

    fake = Manifest.model_validate(payload)

    assert fake.source_attestation_handle != source.attestation_handle
    # The recomputed content hash differs from the original — verify with
    # the original's content_hash as the expected value would fail.
    report = verify_manifest(fake, expected_content_hash=bundle.manifest.content_hash())
    assert report.ok is False
    assert "tampered" in report.reason.lower()


# ---------------------------------------------------------------------------
# Calibration-state requirement guard.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_require_calibration_state_enforces_production_invariant() -> None:
    """A production caller may demand the robot be CALIBRATED."""
    robot = load("cartpole")  # zoo loader yields UNCALIBRATED.
    with pytest.raises(SourceError, match="calibration_state"):
        JaxteritySource.from_robot(robot, require_calibration_state="CALIBRATED")


@pytest.mark.unit
def test_require_calibration_state_passes_for_calibrated_robot() -> None:
    """A calibrated robot (with_provenance(..., calibrated=True)) is accepted."""
    base = load("cartpole")
    calibrated = base.with_provenance(
        ("phase1-recipe", "v0.0.1", "tel-hash"), calibrated=True
    )
    source = JaxteritySource.from_robot(
        calibrated, require_calibration_state="CALIBRATED"
    )
    assert source.attestation_handle == bytes.fromhex(calibrated.attestation_handle)


# ---------------------------------------------------------------------------
# Acceptance 3 (T-101): calibration propagates into the *deployed dynamics*,
# not just the attestation handle. ADR-016 lowers a closed-form cartpole, not
# MJX; the closed-form's scalars are sourced from the robot so a recalibration
# moves the lowered plant too — "one model, one truth" at the parameter level.
# ---------------------------------------------------------------------------

# T-101 lands across two repos; Jaxility's consumer wiring needs a Jaxterity
# that exports ``reduced_params`` (it ships first, per the dependency arrow).
# Skip cleanly on an older installed Jaxterity rather than hard-erroring.
_needs_reduced_params = pytest.mark.skipif(
    not hasattr(
        __import__("jaxterity.zoo.cartpole", fromlist=["reduced_params"]),
        "reduced_params",
    ),
    reason="installed jaxterity predates T-101 (no zoo.cartpole.reduced_params)",
)


@_needs_reduced_params
@pytest.mark.unit
def test_reduced_params_source_canonical_constants() -> None:
    """The uncalibrated zoo robot yields exactly the historical hardcode.

    Guards that wiring the closed-form to ``reduced_params`` is behaviour-
    preserving for the shipped robot: ``(g, mc, mp, L) == (9.81, 1.0, 0.1, 0.5)``,
    the constants the closed-form used before T-101.
    """
    from jaxterity.zoo.cartpole import reduced_params

    p = reduced_params(load("cartpole"))
    assert p == pytest.approx({"g": 9.81, "mc": 1.0, "mp": 0.1, "L": 0.5})


@_needs_reduced_params
@pytest.mark.unit
def test_calibration_propagates_into_deployed_dynamics() -> None:
    """Recalibrating a mass moves the lowered closed-form dynamics, not just the
    handle (T-101).

    Before T-101 the closed-form hardcoded ``mp = 0.1``, so doubling the pole
    mass flipped the attestation handle while the deployed plant stayed put — the
    manifest attested to a model that was not the one running. Now the four
    scalars come from the robot via ``reduced_params``, so the same mutation
    moves both.
    """
    import jax.numpy as jnp
    from jaxterity.zoo.cartpole import reduced_params

    from jaxility.zoo.cartpole import _cartpole_ode

    base = load("cartpole")
    heavy = base.with_parameters({"pole.mass": base.parameters()["pole.mass"] * 2.0})

    # Handle moves (the half that already worked).
    assert base.attestation_handle != heavy.attestation_handle

    # Reduced params move: pole mass doubled, the rest unchanged.
    pb, ph = reduced_params(base), reduced_params(heavy)
    assert pb["mp"] == pytest.approx(0.1)
    assert ph["mp"] == pytest.approx(0.2)
    assert (ph["mc"], ph["L"], ph["g"]) == pytest.approx((pb["mc"], pb["L"], pb["g"]))

    # And the *deployed* closed-form dynamics move — the T-101 fix.
    state = jnp.asarray([0.0, 0.3, 0.0, 0.5])
    control = jnp.asarray([1.0])
    dx_base = _cartpole_ode(pb)(state, control)
    dx_heavy = _cartpole_ode(ph)(state, control)
    assert float(jnp.max(jnp.abs(dx_heavy - dx_base))) > 1e-6


# ---------------------------------------------------------------------------
# T-110: Crazyflie closed-form quadrotor dynamics — a second robot end-to-end.
# The deployed plant is a closed form (ADR-016), sourced from the calibrated
# Robot. These prove it is (a) faithful to the MJX reference, (b) coupled to
# calibration, and (c) inside the smooth-op subset (lowers to CasADi).
# ---------------------------------------------------------------------------


def _has_crazyflie_reference() -> bool:
    try:
        from jaxterity.zoo import crazyflie
    except Exception:
        return False
    return hasattr(crazyflie, "thrust_dynamics")


_needs_crazyflie = pytest.mark.skipif(
    not _has_crazyflie_reference(),
    reason="installed jaxterity predates the Crazyflie zoo entry "
    "(no zoo.crazyflie.thrust_dynamics)",
)


@_needs_crazyflie
@pytest.mark.unit
def test_crazyflie_closed_form_matches_mjx_reference() -> None:
    """The deployed closed form reproduces the MJX reference to ~ULP.

    ADR-016 lowers a closed form, not MJX; this guards that the closed form is a
    faithful stand-in for the real (MJX) dynamics of *this* robot, not merely a
    plausible quadrotor — the faithfulness contract the Cartpole entry also meets.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    from jaxterity.zoo import crazyflie

    from jaxility.zoo.crazyflie import _quadrotor_ode, _reduced_params

    reference = crazyflie.thrust_dynamics()
    closed_form = _quadrotor_ode(_reduced_params(crazyflie.load()))

    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(64):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        state = jnp.asarray(
            np.concatenate(
                [
                    rng.normal(size=3),
                    quat,
                    0.5 * rng.normal(size=3),
                    2.0 * rng.normal(size=3),
                ]
            )
        )
        control = jnp.asarray(
            np.concatenate(
                [
                    [crazyflie.hover_thrust() * (1.0 + 0.3 * rng.normal())],
                    1e-4 * rng.normal(size=3),
                ]
            )
        )
        err = float(
            jnp.max(jnp.abs(reference(state, control) - closed_form(state, control)))
        )
        max_err = max(max_err, err)
    assert max_err < 1e-9, f"closed form drifts from MJX by {max_err:g}"


@_needs_crazyflie
@pytest.mark.unit
def test_crazyflie_calibration_propagates_into_deployed_dynamics() -> None:
    """Doubling the vehicle mass moves both the attestation handle and the
    lowered closed-form dynamics — one model, one truth (T-110, mirrors T-101).
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from jaxterity.zoo import crazyflie

    from jaxility.zoo.crazyflie import _quadrotor_ode, _reduced_params

    base = crazyflie.load()
    heavy = base.with_parameters({"cf2.mass": base.parameters()["cf2.mass"] * 2.0})

    # Handle moves (the half that already worked).
    assert base.attestation_handle != heavy.attestation_handle

    pb, ph = _reduced_params(base), _reduced_params(heavy)
    assert ph["m"] == pytest.approx(2.0 * pb["m"])
    assert ph["I"] == pytest.approx(pb["I"])

    # And the *deployed* closed form moves: at identity attitude (rest) a thrust
    # of 2·(base weight) gives +g of vertical accel under the base mass but ~0
    # under the doubled mass.
    state = jnp.zeros(13).at[3].set(1.0)  # identity quaternion (w=1), at rest
    control = jnp.asarray([2.0 * crazyflie.hover_thrust(), 0.0, 0.0, 0.0])
    dx_base = _quadrotor_ode(pb)(state, control)
    dx_heavy = _quadrotor_ode(ph)(state, control)
    assert float(jnp.max(jnp.abs(dx_heavy - dx_base))) > 1e-6


@_needs_crazyflie
@pytest.mark.unit
def test_crazyflie_closed_form_lowers_to_casadi() -> None:
    """The closed form is inside the smooth-op subset acados consumes: it
    translates to a CasADi function (no CoverageError), so Crazyflie is
    end-to-end lowerable, not merely simulatable.
    """
    from jaxility.lowering import CasadiFunction, translate
    from jaxility.zoo.crazyflie import _dynamics_factory

    dynamics, state_shape, control_shape = _dynamics_factory()
    casadi_fn = translate(
        dynamics,
        in_shapes=(state_shape, control_shape),
        dtype="float64",
        target_family="mock-cortex-a",
        name="crazyflie",
    )
    assert isinstance(casadi_fn, CasadiFunction)
