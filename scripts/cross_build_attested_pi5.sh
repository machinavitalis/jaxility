#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
#
# Reproducible attested cross-build for the Pi 5 controller artifact (T-037).
#
# Produces a Cartpole controller built with the *pinned* Arm GNU 15.2.Rel1
# toolchain (so the manifest records a reproducible toolchain, invariant 3),
# rather than the Pi's native gcc (the T-034 build-on-Pi proof path).
#
# Flow (host = Linux or macOS with Docker; the toolchain runs in the container,
# which is why the pinned toolchain need not exist on the host):
#
#   1. Build the cross-build image (pinned toolchain):
#        docker build -f docker/cross-aarch64.Dockerfile -t jaxility-cross-aarch64 .
#   2. Stage the acados source + this repo's CMake toolchain file into $STAGE.
#   3. Cross-build acados/blasfeo/hpipm as STATIC archives with the pinned
#      toolchain (BUILD_SHARED_LIBS=OFF), matching jaxility.builder_deps.
#   4. Generate the controller C on the host (jaxility.bench.cartpole +
#      jaxility.hil/bench generators) and stage it under $STAGE/controller.
#   5. Cross-compile the controller (HIL + benchmark binaries) against the
#      static deps with the pinned A76 flags.
#
# The resulting binaries link acados statically; only libc/libm are dynamic
# and they reference GLIBC <= 2.34 (<= the Pi's 2.36), so they load on the Pi.
# Deploy + verify with jaxility.hil / jaxility.bench (an SshRunner at the Pi),
# and build the attested Manifest recording `aarch64-none-linux-gnu-gcc 15.2.1`
# + the dep-archive BLAKE3 hashes (see jaxility.builder_cross.cross_build_for_target
# for the canonical manifest shape).
#
# This script automates steps 3 + 5 (the container cross-build); steps 2 and 4
# (host staging) are environment-specific and shown above.
set -euo pipefail

IMAGE="${JAXILITY_CROSS_IMAGE:-jaxility-cross-aarch64}"
STAGE="${JAXILITY_CROSS_STAGE:-/tmp/xbuild}"
MODEL="${JAXILITY_MODEL:-cartpole_lqr}"
BLASFEO_TARGET="${JAXILITY_BLASFEO_TARGET:-ARMV8A_ARM_CORTEX_A76}"
JOBS="${JAXILITY_JOBS:-4}"

test -d "${STAGE}/acados" || { echo "stage ${STAGE}/acados (acados source) first" >&2; exit 1; }
test -d "${STAGE}/controller" || { echo "stage ${STAGE}/controller (generated C) first" >&2; exit 1; }
test -f "${STAGE}/aarch64-none-linux-gnu.cmake" \
  || cp cmake/toolchains/aarch64-none-linux-gnu.cmake "${STAGE}/aarch64-none-linux-gnu.cmake"

docker run --rm -v "${STAGE}:/work" "${IMAGE}" bash -euo pipefail -c "
  # 3. Static deps (acados/blasfeo/hpipm) with the pinned cross toolchain.
  if [ ! -f /work/deps-prefix/lib/libacados.a ]; then
    cmake -S /work/acados -B /work/deps-build \
      -DCMAKE_TOOLCHAIN_FILE=/work/aarch64-none-linux-gnu.cmake \
      -DACADOS_INSTALL_DIR=/work/deps-prefix -DCMAKE_INSTALL_PREFIX=/work/deps-prefix \
      -DBUILD_SHARED_LIBS=OFF -DBLASFEO_TARGET=${BLASFEO_TARGET} \
      -DACADOS_WITH_QPOASES=OFF -DACADOS_SILENT=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build /work/deps-build --parallel ${JOBS} --target install
  fi

  # 5. Cross-compile the controller (HIL + benchmark) against the static deps.
  P=/work/deps-prefix; G=/work/controller; M=${MODEL}
  INC=\"-I\$G -I\$P/include -I\$P/include/acados -I\$P/include/blasfeo/include -I\$P/include/hpipm/include\"
  SRCS=\"\$G/acados_solver_\$M.c \$G/acados_sim_solver_\$M.c \$G/\${M}_model/\"*.c
  GRP=\"-Wl,--start-group \$P/lib/libacados.a \$P/lib/libhpipm.a \$P/lib/libblasfeo.a -Wl,--end-group -lm -lpthread\"
  for which in hil bench; do
    aarch64-none-linux-gnu-gcc -mcpu=cortex-a76 -O3 -std=c99 -Wno-unused-parameter \
      \$INC \$G/\${M}_\${which}_main.c \$SRCS \$GRP -o /work/\${M}_\${which}
    echo \"built \${M}_\${which}\"
  done
  echo 'glibc symbol versions required (must be <= the target glibc):'
  aarch64-none-linux-gnu-objdump -T /work/\${M}_hil | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tr '\n' ' '; echo
"
echo "artifacts: ${STAGE}/${MODEL}_hil  ${STAGE}/${MODEL}_bench"
