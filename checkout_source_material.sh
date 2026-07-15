#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [WORKSPACE_ROOT]

Clone the BMG programming reference repositories into WORKSPACE_ROOT.
The default is the parent directory of this repository, so the resulting
layout is:

  WORKSPACE_ROOT/
    xpu_playground/
    llvm/
    xetla/
    level-zero-spec/
EOF
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required" >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root=${1:-"$(dirname -- "$script_dir")"}
mkdir -p -- "$workspace_root"
workspace_root=$(cd -- "$workspace_root" && pwd)

checkout_repository() {
  local name=$1
  local url=$2
  local branch=$3
  local revision=$4
  local destination="$workspace_root/$name"

  echo "[$name]"

  if [[ ! -e $destination ]]; then
    echo "  cloning $url"
    git clone --filter=blob:none --depth 1 --single-branch \
      --branch "$branch" -- "$url" "$destination"
  elif [[ ! -d $destination/.git ]]; then
    echo "error: $destination exists but is not a Git repository" >&2
    return 1
  fi

  local current_revision
  current_revision=$(git -c safe.directory="$destination" -C "$destination" \
    rev-parse HEAD 2>/dev/null || true)
  if [[ $current_revision == "$revision" ]]; then
    echo "  already at $revision"
    return 0
  fi

  if [[ -n $(git -c safe.directory="$destination" -C "$destination" \
    status --porcelain) ]]; then
    echo "error: $destination has local changes and is not at $revision" >&2
    echo "       preserve or remove those changes before retrying" >&2
    return 1
  fi

  if ! git -c safe.directory="$destination" -C "$destination" \
    cat-file -e "$revision^{commit}" 2>/dev/null; then
    echo "  fetching pinned revision $revision"
    git -c safe.directory="$destination" -C "$destination" fetch \
      --filter=blob:none --depth 1 --no-tags -- "$url" "$revision"
  fi

  git -c safe.directory="$destination" -C "$destination" \
    checkout --detach "$revision"
  echo "  checked out $revision"
}

checkout_repository \
  llvm \
  https://github.com/intel/llvm.git \
  sycl \
  0ef90649443bdc833902a5fadbf4daf9bfe992d8

checkout_repository \
  xetla \
  https://github.com/intel/xetla.git \
  main \
  7a1acbde4ff608141500e142324923257605862a

checkout_repository \
  level-zero-spec \
  https://github.com/oneapi-src/level-zero-spec.git \
  master \
  60c28d2051071fdd484a72306c43fa0519a515b0

echo "Source material is ready under $workspace_root"
echo "Apply the XeTLA oneAPI 2026 patch with:"
echo "  git -C '$workspace_root/xetla' apply '$script_dir/patches/xetla-oneapi-2026.patch'"
