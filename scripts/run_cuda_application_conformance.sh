#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_dir="${1:-${source_dir}/build/cuda-application-${timestamp}}"
legacy_root="${2:-${CM_LEGACY_ROOT:-}}"
legacy_commit="4896f543c6250f053eea2312e628cc3a96bf7408"

if [[ -e "${report_dir}" ]]; then
  printf 'report path already exists: %s\n' "${report_dir}" >&2
  exit 2
fi
mkdir -p "${report_dir}"

finish() {
  status=$?
  trap - EXIT
  if [[ ${status} -eq 0 ]]; then
    result=pass
  else
    result=fail
  fi
  {
    printf 'result\t%s\n' "${result}"
    printf 'exit_code\t%s\n' "${status}"
    printf 'completed_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${report_dir}/result.tsv"
  (
    cd "${report_dir}"
    find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort | while IFS= read -r file; do
      cmake -E sha256sum "${file}"
    done
  ) >"${report_dir}/SHA256SUMS"
  printf 'CUDA application conformance %s; evidence: %s\n' "${result}" "${report_dir}"
  exit "${status}"
}
trap finish EXIT

for command_name in cmake git ninja nvcc nvidia-smi uv; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'required command is unavailable: %s\n' "${command_name}" >&2
    exit 2
  fi
done

if [[ "$(uname -s)" != Linux ]]; then
  printf 'CUDA application conformance requires Linux\n' >&2
  exit 2
fi
if [[ -z "${legacy_root}" || ! -d "${legacy_root}/.git" ]]; then
  printf 'a pinned CellModeller checkout is required as the second argument\n' >&2
  exit 2
fi
actual_legacy_commit="$(git -C "${legacy_root}" rev-parse HEAD)"
if [[ "${actual_legacy_commit}" != "${legacy_commit}" ]]; then
  printf 'legacy checkout must be at %s, found %s\n' \
    "${legacy_commit}" "${actual_legacy_commit}" >&2
  exit 2
fi

cd "${source_dir}"
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'CUDA application conformance requires a clean worktree\n' >&2
  git status --short >&2
  exit 2
fi
if ! nvidia-smi -L >"${report_dir}/nvidia-devices.txt"; then
  printf 'nvidia-smi could not discover an NVIDIA device\n' >&2
  exit 2
fi
if [[ ! -s "${report_dir}/nvidia-devices.txt" ]]; then
  printf 'nvidia-smi reported no NVIDIA devices\n' >&2
  exit 2
fi

{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'legacy_commit\t%s\n' "${actual_legacy_commit}"
  printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host\t%s\n' "$(hostname)"
  printf 'kernel\t%s\n' "$(uname -a)"
  printf 'uv\t%s\n' "$(uv --version)"
  printf 'nvcc_release\t%s\n' "$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -n 1)"
} >"${report_dir}/environment.tsv"
nvcc --version >"${report_dir}/nvcc.txt"
nvidia-smi \
  --query-gpu=index,uuid,name,compute_cap,driver_version,memory.total,pci.bus_id \
  --format=csv,noheader,nounits >"${report_dir}/gpus.csv"
nvidia-smi -q >"${report_dir}/nvidia-smi.txt"

export CMAKE_ARGS="-DCM_ENABLE_CUDA=ON -DCM_ENABLE_METAL=OFF -DCM_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=native"
uv sync --locked --all-extras --reinstall-package microsimulator \
  2>&1 | tee "${report_dir}/python-build.log"
uv run microsimulator devices --json >"${report_dir}/devices.json"
uv run python -c \
  'from microsimulator import BackendKind, backend_device_count; assert backend_device_count(BackendKind.CUDA) > 0' \
  2>&1 | tee "${report_dir}/cuda-runtime.log"

CM_LEGACY_ROOT="${legacy_root}" uv run pytest -q \
  --junitxml="${report_dir}/pytest.xml" 2>&1 | tee "${report_dir}/pytest.log"
uv run python scripts/run_legacy_example_matrix.py \
  --legacy-root "${legacy_root}" \
  --backend cpu \
  --backend cuda \
  --output "${report_dir}/legacy-example-matrix.json" \
  2>&1 | tee "${report_dir}/legacy-example-matrix.log"
