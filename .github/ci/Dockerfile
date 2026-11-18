# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git ninja-build && rm -rf /var/lib/apt/lists/*

ENV CARGO_HOME=/opt/gptqmodel/cargo \
    RUSTUP_HOME=/opt/gptqmodel/rustup
ENV PATH=$CARGO_HOME/bin:$PATH
COPY .github/scripts/ci_install_uv.sh .github/scripts/ci_install_rust.sh /tmp/ci-bootstrap/
RUN bash /tmp/ci-bootstrap/ci_install_uv.sh /usr/local/bin \
    && bash /tmp/ci-bootstrap/ci_install_rust.sh \
    && rm -rf /tmp/ci-bootstrap

WORKDIR /workspace
