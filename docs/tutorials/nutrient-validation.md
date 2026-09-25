# Nutrient penetration and attached-population growth

Run the controlled numerical study with:

For backend selection, PowerShell syntax, quoted JSON parameters, and paths with spaces, see [tutorial commands by backend and shell](commands.md#choose-a-shell). Multiline commands on this page use POSIX shell backslashes; the guide provides the PowerShell equivalents and [explicit CPU, Metal, and CUDA trap launches](commands.md#run-the-same-trap-on-cpu-metal-or-cuda).

```console
uv run python scripts/run_nutrient_benchmarks.py --backend cpu --output build/nutrient-cpu.json
uv run python scripts/run_nutrient_benchmarks.py --backend metal --output build/nutrient-metal.json
```

The experiment places 45 fixed cells in a 20 x 40 x 4 micrometer channel. Initially nutrient is absent; inlet concentration is one and outlet concentration zero. Diffusion is 4 square micrometers per model time, mean inlet speed 0.3 micrometers per model time, maximum cylindrical-length growth rate 0.2 per model time, Monod constant 0.2, and biomass yield 0.5. Time is an illustrative model scale. There is no division, mechanics, or detachment in this experiment. These choices isolate nutrient delivery, growth, and stationary resistance.

## Conservative coupling

Each attached cell has conserved biochemical amount `B_i = pi*r_i²*(l_i + 2*r_i)`. A separable tent kernel K_i with physical radius 4 micrometers is integrated exactly over voxels and normalized so `sum_j K_ij V_j = 1`. Its support stays fixed as the grid is refined. Nutrient seen by a cell is `cbar_i = sum_j K_ij c_j V_j`.

At step n, define `a_i = mu*pi*r_i²*l_i / (Y*(K_M + cbar_i))`. The grid loss is `lambda_j = sum_i a_i K_ij`. Native backward Euler advances conservative transport with this nonnegative loss, so the uptake assigned to cell i is `U_i = dt*a_i*sum_j K_ij*c_j^(n+1)*V_j`. The cell then gains `Delta B_i = Y*U_i`, with length updated accordingly. This is a first-order, semi-implicit Monod approximation with a lagged denominator and cylinder amount. It avoids the mesh-dependent singularity of a point sink and makes biomass gain equal the implicit nutrient loss to solver and rounding error. The four interactive tutorials separately exercise the native realized-growth rate instruction with explicit cell exchange.

The flow resistance uses the same physical kernel and only attached biomass. Its coefficient is 40 in the normalized shallow closure. Flow is initially loaded and refreshed at the stated physical interval. All runs use the same physical domain, population, kernel, and parameters.

## What is measured

The script compares spacings 2, 1, and 0.5 micrometers, timesteps 0.04 and 0.02, and flow-refresh intervals 0.4 and 0.2 over eight model time units. It records total biomass gain, the first downstream crossing of half the inlet nutrient concentration, spatial growth, and the balance `nutrient remaining + biomass gained / yield = net boundary supply`. Boundary supply uses the engine's discrete ghost-center concentration convention and backward-Euler end-of-step fluxes. The chosen inlet boundary is therefore also refined with the grid.

The JSON stores parameters, source commit, every profile, sensitivity comparisons, and pass/fail gates. CPU and Metal development runs gave about 31.00 biomass-volume units gained and a 5.71 micrometer half-concentration depth at spacing 1. Halving spacing changed biomass gain by 2.8% and penetration by 3.5%; halving the timestep changed gain by 0.44%; halving flow-refresh time changed gain by less than 0.001%. Maximum nutrient-balance error was below 0.008% of net supply. Regenerate these figures from the recorded commit before using them as release evidence.

These results support a numerical demonstration of spatially limited growth. They do not calibrate nutrient yield, physical time, or the resistance law. The weak refresh sensitivity is specific to this slowly changing attached population, not a universal refresh recommendation. First-order upwind numerical diffusion remains: at spacing 1, its nominal scale `U*h/2 = 0.15` is about 3.8% of physical diffusion; it halves with spacing. More advective applications need their own refinement study or a higher-order transport method.
