# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
#
# Reproducible cross-build environment for the attested Pi 5 artifact (T-037).
#
# Provides a Linux host with the *pinned* Arm GNU 15.2.Rel1 toolchain so
# `jaxility.builder_cross.cross_build_for_target` produces an artifact whose
# manifest records a reproducible toolchain (invariant 3) — unlike the
# build-on-Pi proof path (T-034), which records the Pi's native gcc.
#
# This image is arm64 (native on Apple Silicon / Arm CI; the toolchain is the
# aarch64-host build of the same `aarch64-none-linux-gnu` cross compiler the
# x86_64 CI job uses — the *target* and version are identical, so the recorded
# `aarch64-none-linux-gnu-gcc 15.2.1` pin matches either host).
#
# Build:
#   docker build -f docker/cross-aarch64.Dockerfile -t jaxility-cross-aarch64 .
# The Python ML stack (jax / casadi / acados-template) and the acados source
# tree are mounted/installed at run time, not baked in, to keep the image lean
# and the toolchain layer cached.
FROM --platform=linux/arm64 ubuntu:24.04

ARG ARM_GNU_VERSION=15.2.rel1
ARG ARM_GNU_TARBALL=arm-gnu-toolchain-15.2.rel1-aarch64-aarch64-none-linux-gnu.tar.xz
ARG ARM_GNU_URL=https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/arm-gnu-toolchain-15.2.rel1-aarch64-aarch64-none-linux-gnu.tar.xz

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils cmake make git \
        python3 python3-pip python3-venv build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned Arm GNU toolchain (T-031 / PI5 ToolchainPin). Same version + target
# triple as the CI x86_64-host install; aarch64-host build for native speed.
RUN mkdir -p /opt/arm-gnu-toolchain-${ARM_GNU_VERSION} \
    && curl --fail --silent --show-error --location \
        -o /tmp/arm-gnu.tar.xz "${ARM_GNU_URL}" \
    && tar -xJf /tmp/arm-gnu.tar.xz \
        -C /opt/arm-gnu-toolchain-${ARM_GNU_VERSION} --strip-components=1 \
    && rm /tmp/arm-gnu.tar.xz

ENV PATH=/opt/arm-gnu-toolchain-${ARM_GNU_VERSION}/bin:${PATH}

# Fail the build loudly if the pin is wrong (matches verify_toolchain_installed).
RUN aarch64-none-linux-gnu-gcc --version | head -1 \
    && aarch64-none-linux-gnu-gcc --version | grep -q "15.2.1"

WORKDIR /work
