# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Multi-contact centroidal MPC template: friction-box bounds, lowering, deploy."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxility.lowering import OcpTemplateSpec, translate
from jaxility.templates import friction_box_bounds, multi_contact_centroidal_mpc

_MASS = 40.0
_P0 = jnp.array([0.1, 0.05, 0.0])
_P1 = jnp.array([-0.1, -0.05, 0.0])
_G = jnp.array([0.0, 0.0, -9.81])


def _srbd2(state, u):
    """Two-contact single-rigid-body dynamics (matches
    ``jaxterity.locomotion.reduced.srbd_dynamics`` with two baked contact points).

    state = ``[c(3), ċ(3), L(3)]``;  u = ``[f0(3), f1(3)]`` (static slices, no
    reshape, so the graph stays inside jaxility's supported smooth-op subset)."""
    c = state[0:3]
    cdot = state[3:6]
    f0 = u[0:3]
    f1 = u[3:6]
    cddot = (f0 + f1) / _MASS + _G
    Ldot = jnp.cross(_P0 - c, f0) + jnp.cross(_P1 - c, f1)
    return jnp.concatenate([cdot, cddot, Ldot])


@pytest.fixture(scope="module")
def srbd_cf():
    return translate(_srbd2, in_shapes=((9,), (6,)), name="srbd2")


# --- friction-box bounds ---------------------------------------------------


@pytest.mark.unit
def test_friction_box_double_support():
    lo, hi = friction_box_bounds((True, True), fz_max=200.0, mu=0.7)
    assert len(lo) == 6 and len(hi) == 6
    # each stance foot: fz in [0, 200]; |fx|,|fy| <= 140
    assert lo == (-140.0, -140.0, 0.0, -140.0, -140.0, 0.0)
    assert hi == (140.0, 140.0, 200.0, 140.0, 140.0, 200.0)


@pytest.mark.unit
def test_friction_box_swing_foot_pinned_to_zero():
    lo, hi = friction_box_bounds((True, False), fz_max=200.0, mu=0.7)
    # foot 0 (stance) bounded; foot 1 (swing) forced to zero force.
    assert hi[0:3] == (140.0, 140.0, 200.0)
    assert lo[3:6] == (0.0, 0.0, 0.0) and hi[3:6] == (0.0, 0.0, 0.0)


# --- spec construction -----------------------------------------------------


@pytest.mark.unit
def test_multi_contact_spec_shape(srbd_cf):
    spec = multi_contact_centroidal_mpc(
        srbd_cf,
        Q=(10.0,) * 9,
        R=(0.01,) * 6,
        initial_com_state=(0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        contact_active=(True, True),
    )
    assert isinstance(spec, OcpTemplateSpec)
    assert len(spec.state_cost) == 9  # SRBD includes angular momentum
    assert len(spec.input_cost) == 6  # two stacked 3-D contact forces
    # double-support friction box propagated to the input bounds.
    assert spec.input_upper == (140.0, 140.0, 200.0, 140.0, 140.0, 200.0)


@pytest.mark.unit
def test_multi_contact_swing_zeroes_force_bounds(srbd_cf):
    spec = multi_contact_centroidal_mpc(
        srbd_cf,
        Q=(1.0,) * 9,
        R=(0.1,) * 6,
        initial_com_state=(0.0,) * 9,
        contact_active=(True, False),  # right foot swinging
    )
    assert spec.input_lower[3:6] == (0.0, 0.0, 0.0)
    assert spec.input_upper[3:6] == (0.0, 0.0, 0.0)


@pytest.mark.unit
def test_srbd_lowers_to_supported_ops(srbd_cf):
    """The SRBD plant lowers cleanly (translate would raise on an unsupported op),
    so it is deployable through the jaxility pipeline."""
    assert srbd_cf.input_shapes == ((9,), (6,))
    assert isinstance(srbd_cf.primitives_used, frozenset)
    assert srbd_cf.primitives_used  # non-empty: real ops were lowered


# --- deployability: cross-compile the lowered SRBD for Cortex-M -------------


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("arm-none-eabi-gcc") is None, reason="arm-none-eabi-gcc required"
)
def test_srbd_cross_compiles_for_cortex_m(srbd_cf):
    """CasADi codegen of the lowered SRBD cross-compiles to a Cortex-M4 object —
    proving the reduced model is embeddable (the acados OCP build is separate and
    needs acados; this validates the dynamics-lowering half end to end)."""
    work = Path(tempfile.mkdtemp(prefix="srbd_m4_"))
    cwd = Path.cwd()
    import os

    os.chdir(work)
    try:
        c_name = srbd_cf.fn.generate("srbd2.c")
        flags = [
            "-mcpu=cortex-m4",
            "-mthumb",
            "-mfpu=fpv4-sp-d16",
            "-mfloat-abi=hard",
            "-O3",
            "-c",
        ]
        r = subprocess.run(
            ["arm-none-eabi-gcc", *flags, c_name, "-o", "srbd2.o"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        elf = subprocess.run(
            ["arm-none-eabi-readelf", "-h", "srbd2.o"], capture_output=True, text=True
        ).stdout
        assert "ELF32" in elf and "ARM" in elf
    finally:
        os.chdir(cwd)


# --- numeric sanity: matches the jaxterity reference SRBD -------------------


@pytest.mark.unit
def test_srbd_matches_reference_dynamics(srbd_cf):
    """The lowered CasADi function reproduces the JAX SRBD (lowering is exact)."""
    rng = np.random.default_rng(0)
    state = jnp.asarray(rng.normal(size=9))
    u = jnp.asarray(rng.normal(size=6))
    jax_out = np.asarray(_srbd2(state, u))
    cas_out = np.asarray(srbd_cf.fn(state, u)).ravel()
    np.testing.assert_allclose(cas_out, jax_out, atol=1e-9)
