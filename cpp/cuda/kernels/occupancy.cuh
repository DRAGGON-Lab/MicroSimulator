#pragma once
#include <cuda_runtime.h>

#include "core/occupancy_device.hpp"

namespace cm::cuda {
void launch_occupancy(detail::OccupancyKernel kernel, const detail::OccupancyParameters& parameters,
                      void* const* buffers, cudaStream_t stream);
}
