# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""acados problem templates: LQR, trajectory-tracking MPC, WBC, centroidal MPC.

Templates parametrise over translated dynamics, horizon, sample rate,
and cost weights and produce an :class:`OcpTemplateSpec` that
:func:`jaxility.lowering.build_ocp` consumes. Templates land
incrementally:

* :func:`lqr` (T-022) — quadratic-stabilisation MPC.
* ``tracking_mpc`` (T-023) — trajectory-tracking MPC.
* ``wbc`` (T-024) — task-space whole-body control.
* ``centroidal_mpc`` (T-025) — humanoid centroidal MPC.
"""

from .centroidal_mpc import (
    centroidal_mpc,
    friction_box_bounds,
    multi_contact_centroidal_mpc,
)
from .lqr import lqr
from .tracking_mpc import set_reference_trajectory, tracking_mpc
from .wbc import WBC_SCHEMA_V0, WBCTask, wbc

__all__ = [
    "WBC_SCHEMA_V0",
    "WBCTask",
    "centroidal_mpc",
    "friction_box_bounds",
    "multi_contact_centroidal_mpc",
    "lqr",
    "set_reference_trajectory",
    "tracking_mpc",
    "wbc",
]
