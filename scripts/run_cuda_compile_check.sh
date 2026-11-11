#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_dir="${1:-${source_dir}/build/cuda-compile-${timestamp}}"
container_image="${MICROSIMULATOR_CUDA_CONTAINER_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04}"
cuda_architectures="${MICROSIMULATOR_CUDA_ARCHITECTURES:-75}"

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
  printf 'CUDA compile check %s; evidence: %s\n' "${result}" "${report_dir}"
  exit "${status}"
}
trap finish EXIT

for command_name in cmake docker git; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'required command is unavailable: %s\n' "${command_name}" >&2
    exit 2
  fi
done

cd "${source_dir}"
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'CUDA compile check requires a clean worktree\n' >&2
  git status --short >&2
  exit 2
fi

if ! docker image inspect "${container_image}" >/dev/null 2>&1; then
  docker pull "${container_image}" 2>&1 | tee "${report_dir}/pull.log"
fi

{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host\t%s\n' "$(hostname)"
  printf 'container_image\t%s\n' "${container_image}"
  printf 'cuda_architectures\t%s\n' "${cuda_architectures}"
  printf 'docker_image_id\t%s\n' "$(docker image inspect --format '{{.Id}}' "${container_image}")"
} >"${report_dir}/environment.tsv"

docker version >"${report_dir}/docker.txt"
docker image inspect "${container_image}" >"${report_dir}/image.json"

docker run --rm \
  --mount "type=bind,src=${source_dir},dst=/source,readonly" \
  --env "MICROSIMULATOR_CUDA_ARCHITECTURES=${cuda_architectures}" \
  "${container_image}" \
  bash -lc '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends cmake ninja-build
    uname -a
    cmake --version
    ninja --version
    g++ --version
    nvcc --version
    cmake -S /source -B /build -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CUDA_ARCHITECTURES="${MICROSIMULATOR_CUDA_ARCHITECTURES}" \
      -DMICROSIMULATOR_BUILD_PYTHON=OFF \
      -DMICROSIMULATOR_BUILD_TESTS=ON \
      -DMICROSIMULATOR_ENABLE_CUDA=ON \
      -DMICROSIMULATOR_ENABLE_METAL=OFF
    cmake --build /build
    ctest --test-dir /build -N
  ' 2>&1 | tee "${report_dir}/compile.log"
