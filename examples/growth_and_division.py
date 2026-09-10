from microsimulator import CellInit, Simulation, Vec3

simulation = Simulation()
cell = CellInit()
cell.position = Vec3(0.0, 0.0, 0.0)
cell.length = 4.0
cell.radius = 0.5
cell.growth_rate = 0.2

parent = simulation.add_cell(cell)
simulation.step(0.5)
first, second = simulation.divide_equal(parent)

for daughter in simulation.cells():
    print(
        f"cell={daughter.id} parent={simulation.lineage_parent(daughter.id)} "
        f"slot={daughter.slot} x={daughter.position.x:.3f} length={daughter.length:.3f}"
    )

assert {first, second} == {daughter.id for daughter in simulation.cells()}
