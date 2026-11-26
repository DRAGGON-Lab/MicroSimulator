# ADR 0028: limited second-order advection

- Status: accepted
- Date: 2026-11-26

## Context

Signal transport takes each face's advective flux from the upwind site. That is first order: it never produces a value outside the range of its neighbors, and it carries a numerical diffusion of about half the speed times the spacing. Where flow runs along a grid axis and the liquids lie side by side, there is nothing for it to smear at a steady state. Where flow crosses the grid at an angle, as in a bend or where channels meet, it blends neighboring streams whatever their real diffusivity. In a trap, with weak flow and strong diffusion, that is harmless. In a device whose question is how far two liquids have mixed, it answers the question itself.

## Decision

A signal grid names its advection scheme: `upwind`, the default, or `van_leer`.

Under `van_leer` a face's flux is the upwind flux plus `u * 0.5 * psi(r) * (c_D - c_U)`, where `c_U` and `c_D` are the face's upwind and downwind sites, `r = (c_U - c_UU) / (c_D - c_U)` takes the site two upwind, and `psi(r) = (r + |r|) / (1 + |r|)` is the van Leer limiter. The implementation evaluates the added concentration as `b * a / (b + a)` with `b = c_U - c_UU` and `a = c_D - c_U` when the two agree in sign, and zero otherwise, which is the same quantity without a division by a vanishing difference. Where the field is smooth the scheme is second order. At a front the limiter falls toward upwind, and at an extremum it is upwind.

A face takes the correction only when its upwind site, its downwind site and the site two upwind are all fluid sites of the grid joined by open faces. Beside a wall, a closed face or a boundary reservoir, and on an axis of fewer than three sites, the face keeps the upwind flux. The same three sites decide a face's correction from either side of it, so each face still has one flux and transport stays conservative. A periodic axis wraps.

Time integration:

- **Forward Euler.** The explicit stability check counts advective outflow twice, because a limited face flux carries up to twice the upwind flux.
- **Backward Euler and Crank-Nicolson.** The Jacobi iteration recomputes transport rates from its iterate each sweep and divides by the upwind operator's diagonal, so the correction enters the recomputed rates and the diagonal is unchanged. The correction makes the system nonlinear, and the undamped iteration does not settle; each sweep therefore moves half way to its Jacobi update (`van_leer_relaxation`). Convergence is judged on the freshly evaluated residual, as for upwind.
- **The step it takes.** The damped iteration settles in a few tens of sweeps while one step carries the fastest fluid across about six sites or fewer, slows sharply beyond that, and fails near thirty. Stronger damping does not extend the range. A solve that stops short of its tolerance is reported, never relaxed, and under this scheme the report says to shorten the step.
- **Levels at zero.** Upwind transport of non-negative levels cannot produce a negative one. The limited flux can, by rounding alone, where a level is at zero. After a step under `van_leer`, a level below zero by no more than sixteen units of binary32 rounding of the field's largest level becomes zero; any other negative level is refused as before.

The scheme is part of the grid's specification and of its checkpoint. CPU, Metal and CUDA implement the same correction, damping and settling.

## Validation

- A block carried once around a periodic line by forward Euler keeps its edges within 6 sites and its peak at 1.0, where upwind spreads them over 46 sites and lowers the peak to 0.79. Both conserve the total, and neither leaves the range it began in.
- On a smooth profile the mean error of the rates falls 3.8-fold per halving of the spacing, against 2.0-fold for upwind.
- Beside a reservoir or a wall the rates equal upwind's exactly, and a uniform field has no rate.
- Backward Euler at a step four times the explicit bound converges and stays in range.
- Two streams turned through a right angle leave with a segregation of 0.70, against 0.39 for upwind, where 1 is side by side and 0 is blended.
- Conformance compares each backend with the CPU reference on plain, masked, face-velocity and patched grids for all three integrators, and with cells exchanging on the grid.

## Limits

- The scheme removes the transport's own mixing where flow crosses the grid at an angle. It does not resolve a stream narrower than a site: a result about mixing still needs a spacing that puts several sites across the narrowest stream, and a refinement check.
- The step limit above makes a fine grid with fast flow take many steps. A steady answer does not depend on the step, so the cost is time, not accuracy.
- Backward Euler remains first order in time. A transient front is still spread along the flow by the time integration.
