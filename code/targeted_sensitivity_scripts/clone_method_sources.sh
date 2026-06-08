#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/sources"
mkdir -p "${SRC_DIR}"

clone_or_update() {
  local name="$1"
  local url="$2"
  local dest="${SRC_DIR}/${name}"
  if [ -d "${dest}/.git" ]; then
    git -C "${dest}" fetch --depth 1 origin
  else
    git clone --depth 1 --filter=blob:none "${url}" "${dest}"
  fi
}

sparse_clone_or_update() {
  local name="$1"
  local url="$2"
  local path="$3"
  local dest="${SRC_DIR}/${name}"
  if [ -d "${dest}/.git" ]; then
    git -C "${dest}" fetch --depth 1 origin
  else
    git clone --depth 1 --filter=blob:none --sparse "${url}" "${dest}"
    git -C "${dest}" sparse-checkout set "${path}"
  fi
}

clone_or_update "SSNMDI" "https://github.com/yushanqiu/SSNMDI.git"
clone_or_update "tGPLVM" "https://github.com/architverma1/tGPLVM.git"
echo "Skipping CZI-Latent-Assessment source clone; use scvi-tools or controlled VAE wrapper for revision install audit."
clone_or_update "VASC" "https://github.com/wang-research/VASC.git"
clone_or_update "SAUCIE" "https://github.com/KrishnaswamyLab/SAUCIE.git"
clone_or_update "SCDRHA" "https://github.com/WHY-17/SCDRHA.git"
clone_or_update "scGAE" "https://github.com/ZixiangLuo1161/scGAE.git"
clone_or_update "DREAM" "https://github.com/Crystal-JJ/DREAM.git"
clone_or_update "DRA" "https://github.com/eugenelin1/DRA.git"
clone_or_update "SPDR" "https://github.com/eleozzr/SPDR.git"
clone_or_update "SQuaD-MDS-and-FItSNE-hybrid" "https://github.com/PierreLambert3/SQuaD-MDS-and-FItSNE-hybrid.git"

if command -v git >/dev/null 2>&1; then
  if [ -d "${SRC_DIR}/scvis-dev/.git" ]; then
    git -C "${SRC_DIR}/scvis-dev" fetch --depth 1 origin
  else
    git clone --depth 1 --filter=blob:none "https://bitbucket.org/jerry00/scvis-dev.git" "${SRC_DIR}/scvis-dev"
  fi
fi
