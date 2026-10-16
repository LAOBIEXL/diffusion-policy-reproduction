#!/usr/bin/env bash
set -euo pipefail
export PIP_DEFAULT_TIMEOUT=120
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$(dirname "$0")/.."
PYTHON="${PUSHT_PYTHON:-$HOME/.local/share/diffusion-policy-reproduction/env/bin/python}"
export PATH="$(dirname "$PYTHON"):$PATH"
mkdir -p records/setup records/environment

# Pin the legacy build tools required by Gym 0.21's package metadata.
"$PYTHON" -m pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4 cmake==3.24.3 \
  2>&1 | tee records/setup/build-tools.log

# CUDA 12.8 supplies kernels for this machine's Blackwell GPU.
"$PYTHON" -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  2>&1 | tee records/setup/pytorch.log

"$PYTHON" -m pip install --no-build-isolation -r requirements-pusht.txt \
  2>&1 | tee records/setup/pusht-dependencies.log
"$PYTHON" -m pip install --no-build-isolation --no-deps -e third_party/diffusion_policy \
  2>&1 | tee records/setup/upstream-package.log
"$PYTHON" -m pip check | tee records/setup/pip-check.log
"$PYTHON" -m pip freeze > records/environment/requirements-wsl.lock.txt
