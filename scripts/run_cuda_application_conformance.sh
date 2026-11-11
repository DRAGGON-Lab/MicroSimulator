#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_dir="${1:-${source_dir}/build/cuda-application-${timestamp}}"

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

export CMAKE_ARGS="-DMICROSIMULATOR_ENABLE_CUDA=ON -DMICROSIMULATOR_ENABLE_METAL=OFF -DMICROSIMULATOR_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=native"
uv sync --locked --all-extras --reinstall-package microsimulator \
  2>&1 | tee "${report_dir}/python-build.log"
uv run microsimulator devices --json >"${report_dir}/devices.json"
uv run python -c \
  'from microsimulator import BackendKind, backend_device_count; assert backend_device_count(BackendKind.CUDA) > 0' \
  2>&1 | tee "${report_dir}/cuda-runtime.log"

uv run pytest -q \
  --junitxml="${report_dir}/pytest.xml" 2>&1 | tee "${report_dir}/pytest.log"
