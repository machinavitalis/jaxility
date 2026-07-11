# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
#
# CMake cross-compilation toolchain file for the Raspberry Pi 5 lane
# (Cortex-A76, gnu-linux ABI) — Arm GNU Toolchain 15.2.Rel1
# (``aarch64-none-linux-gnu-*``).
#
# Used by ``jaxility.builder_deps`` to cross-build the acados / blasfeo /
# hpipm static archives that the Pi 5 controller artifact links against
# (T-031 follow-up). The same chain that compiles the acados-generated
# controller C (``jaxility.builder_cross``) compiles its dependencies.
#
# Deliberately ABI/ISA-minimal. We do NOT inject ``-mcpu=cortex-a76``
# here: blasfeo selects its own ``-march=armv8-a+crc+crypto+simd`` per
# its ``ARMV8A_ARM_CORTEX_A76`` target, and Arm GNU 15.2.Rel1 rejects
# ``-mcpu`` alongside ``-march`` (the same constraint documented in
# ``jaxility.builder_cross._FAMILY_CFLAGS``). Letting each component
# pick its own ``-march`` avoids the conflict; the controller artifact
# itself is compiled separately by ``plan_cross_compile`` with the
# A76-tuned flags.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(_jaxility_cross_prefix "aarch64-none-linux-gnu-")
set(CMAKE_C_COMPILER   "${_jaxility_cross_prefix}gcc")
set(CMAKE_CXX_COMPILER "${_jaxility_cross_prefix}g++")
set(CMAKE_ASM_COMPILER "${_jaxility_cross_prefix}gcc")
set(CMAKE_AR           "${_jaxility_cross_prefix}ar"     CACHE FILEPATH "ar")
set(CMAKE_RANLIB       "${_jaxility_cross_prefix}ranlib" CACHE FILEPATH "ranlib")

# Search for libraries/headers in the target rootfs only; run host
# programs (the build itself shells out to no target binaries).
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
