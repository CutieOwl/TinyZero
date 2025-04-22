#!/bin/bash
set -e

# Install dependencies
apt-get update
apt-get install -y \
  liblzma-dev \
  build-essential \
  zlib1g-dev \
  libncurses5-dev \
  libgdbm-dev \
  libnss3-dev \
  libssl-dev \
  libreadline-dev \
  libffi-dev \
  wget \
  curl \
  xz-utils \
  ca-certificates

# Install Python 3.9 from source with lzma support
cd /usr/src
PY_VER=3.9.21
wget https://www.python.org/ftp/python/$PY_VER/Python-$PY_VER.tgz
tar -xzf Python-$PY_VER.tgz
cd Python-$PY_VER
./configure --enable-optimizations
make -j$(nproc)
make altinstall  # Use altinstall to avoid replacing system Python

# Make sure the new Python is used
ln -sf /usr/local/bin/python3.9 /usr/bin/python3.9

# Run the script
export PYTHONPATH=/app
python3.9 /app/data_preproc_cybench.py --local_dir /app/data --from_docker "$@"
EOF

chmod +x /tmp/setup_and_run.sh