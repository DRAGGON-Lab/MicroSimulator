#pragma once

#import <Metal/Metal.h>

namespace microsimulator::metal {

// Safe math with precise functions. The solvers' status checks rely on
// isfinite and on comparisons with NaN, which fast math does not keep.
inline MTLCompileOptions* safe_math_options() {
  MTLCompileOptions* options = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  return options;
}

}  // namespace microsimulator::metal
