# Biomass accounting

The conserved biochemical volume is B = pi r²(l + 2r), with rod centerline length l and radius r. At constant biomass density rho, biomass mass is rho B and an intracellular concentration c represents amount c B. This effective volume is distinct from the geometric capsule volume pi r²l + 4pi r³/3. The latter changes when a parent capsule is replaced by two rounded daughters, so it must not be used interchangeably with B in biomass-dependent feedback.

Native division preserves the parent's outer endpoints and satisfies l1 + l2 = l - 2r. It therefore conserves B exactly in real arithmetic, including unequal division. The division fraction partitions the available cylindrical length, not total biomass. Both daughters inherit concentrations, so intracellular amount is conserved too. Geometry remains a coarse representation of division rather than a volume-preserving model of septum formation.

Growth retains the elongation law dl/dt = g l. Consequently dB/dt = pi r² g l, not g B. `RatePlanBuilder.cell_volume_change_rate()` returns the realized change (B_after - B_before)/dt during the current growth step, with zero at dt=0. A nutrient sink `-rates.cell_volume_change_rate()/yield` therefore consumes exactly the biomass increment divided by yield, up to floating-point error. `rates.cell_volume()` and `microsimulator.biomass.biomass_volume` expose B. A negative concentration is a rejected step, never an invitation to clip away a mass-balance error.

Contact relaxation without explicitly prescribed length increments preserves each rod's length. Numerical overlap corrections therefore cannot manufacture biomass after the biological update. Direct geometry edits and explicitly prescribed mechanical length increments are externally imposed changes; callers must account for their biomass and intracellular amounts.

Colony resistance uses a calibrated biomass-density closure derived from B. Its density must be deposited conservatively over a physical averaging scale independent of voxel size. It is not an exact solid-occupancy reconstruction of individual capsules.
