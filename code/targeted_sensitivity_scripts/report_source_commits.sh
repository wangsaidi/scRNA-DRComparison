#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/sources"

printf "source,commit\n"
for git_dir in "${SRC_DIR}"/*/.git; do
  repo_dir="$(dirname "${git_dir}")"
  name="$(basename "${repo_dir}")"
  commit="$(git -C "${repo_dir}" rev-parse HEAD)"
  printf "%s,%s\n" "${name}" "${commit}"
done
