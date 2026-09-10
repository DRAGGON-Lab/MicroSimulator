from microsimulator import GridShape, SignalGridSpec, Simulation, Vec3

simulation = Simulation()
shape = GridShape()
shape.x = 5
shape.y = 1
shape.z = 1

grid = SignalGridSpec()
grid.signal_count = 1
grid.shape = shape
grid.origin = Vec3(-2.0, 0.0, 0.0)
grid.spacing = Vec3(1.0, 1.0, 1.0)
grid.diffusion = [0.5]
grid.advection = [Vec3()]
simulation.configure_signal_grid(grid, [0.0, 0.0, 1.0, 0.0, 0.0])

for _ in range(4):
    simulation.step(0.25)

print(simulation.signal_levels)
print(simulation.sample_signals(Vec3(-0.5, 12.0, -4.0)))
