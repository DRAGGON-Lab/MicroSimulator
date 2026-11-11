// Opening fragment of every Metal shader source. A Metal library is compiled
// from source at runtime, with no include path, so the build concatenates this
// fragment, the shared definition lists it gives meaning to, and the shared
// structure layouts ahead of each kernel source.

#include <metal_stdlib>

using namespace metal;

#define MICROSIMULATOR_SHARED_FLOAT(name, value) constant float name = value##f;
#define MICROSIMULATOR_SHARED_UINT(name, value) constant uint name = value##u;
#define MICROSIMULATOR_RATE_OP(name, code) constant uint rate_op_##name = code##u;
