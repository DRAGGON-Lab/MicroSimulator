#pragma once

#include "cpu/kernels/contact_solve.hpp"

namespace microsimulator::detail {

// The contact solve of a resident substep on the host, from the
// multipliers in `bindings.multipliers` into `bindings.solve_state`, as a
// device solves it.
void solve_resident_contacts_cpu(const device::ContactStepBindings& bindings,
                                 const device::ContactStepParams& params);

}  // namespace microsimulator::detail
