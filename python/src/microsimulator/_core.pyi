from collections.abc import Mapping
from enum import Enum
from typing import overload

from .culture import CultureState
from .growth import CellGrowth
from .stokes import StokesFlow
from .transport import SoluteTransport

class OccupancyCapsule:
    def __init__(
        self,
        center: tuple[float, float, float],
        direction: tuple[float, float, float],
        length: float,
        radius: float,
    ) -> None: ...

class OccupancyFace:
    first: int
    second: int
    conductance: float
    volume_flux: float
    def __init__(self, first: int, second: int, conductance: float, volume_flux: float) -> None: ...

class OccupancyReservoir:
    def __init__(
        self, site: int, concentration: float, conductance: float, volume_flux: float
    ) -> None: ...

class OccupancyBalance:
    before: float
    after: float
    source: float
    reaction: float
    boundary: float

class OccupancyStep:
    amount: list[float]
    balance: OccupancyBalance
    iterations: int
    relative_residual: float

class OccupancySolver:
    def __init__(self, backend: BackendKind, device_index: int, epsilon_cutoff: float) -> None: ...
    def geometric_porosity(
        self,
        centers: list[tuple[float, float, float]],
        spacing: tuple[float, float, float],
        cells: list[OccupancyCapsule],
        subdivisions: int,
        walls: list[int],
    ) -> list[float]: ...
    def accessible_volumes(self, porosity: list[float], voxel_volume: float) -> list[float]: ...
    def concentration(self, amount: list[float], volume: list[float]) -> list[float]: ...
    def porosity_face(
        self,
        first: int,
        second: int,
        epsilon_first: float,
        epsilon_second: float,
        diffusion: float,
        area: float,
        distance: float,
        intrinsic_velocity: float,
    ) -> OccupancyFace: ...
    def remap_amounts(
        self,
        amount: list[float],
        old_volume: list[float],
        new_volume: list[float],
        neighbors: list[tuple[int, int]],
    ) -> list[float]: ...
    def exchange_weights(self, kernel: list[float], volume: list[float]) -> list[float]: ...
    def backward_euler(
        self,
        amount: list[float],
        volume: list[float],
        faces: list[OccupancyFace],
        dt: float,
        source: list[float],
        loss: list[float],
        reservoirs: list[OccupancyReservoir],
        max_iterations: int,
        relative_tolerance: float,
    ) -> OccupancyStep: ...

class BackendKind(Enum):
    CPU: BackendKind
    METAL: BackendKind
    CUDA: BackendKind

class BackendFeature(Enum):
    GROWTH: BackendFeature
    SPECIES: BackendFeature
    CELL_CONTACTS: BackendFeature
    CELL_MECHANICS: BackendFeature
    EXTERNAL_CONSTRAINTS: BackendFeature
    SIGNALS: BackendFeature
    COUPLED_RATES: BackendFeature
    DEPTH_AVERAGED_FLOW: BackendFeature
    RESOLVED_FLOW: BackendFeature
    CULTURE: BackendFeature

class FlowAxis(Enum):
    X: FlowAxis
    Y: FlowAxis
    Z: FlowAxis

class GridBoundaryKind(Enum):
    NO_FLUX: GridBoundaryKind
    PERIODIC: GridBoundaryKind
    FIXED: GridBoundaryKind

class SignalIntegrationKind(Enum):
    FORWARD_EULER: SignalIntegrationKind
    CRANK_NICOLSON: SignalIntegrationKind
    BACKWARD_EULER: SignalIntegrationKind

class RateOp(Enum):
    CONSTANT: RateOp
    SPECIES: RateOp
    POSITION_X: RateOp
    POSITION_Y: RateOp
    POSITION_Z: RateOp
    CELL_LENGTH: RateOp
    CELL_RADIUS: RateOp
    GROWTH_RATE: RateOp
    CELL_TYPE: RateOp
    CELL_VOLUME: RateOp
    CELL_VOLUME_CHANGE_RATE: RateOp
    CELL_SURFACE_AREA: RateOp
    ADD: RateOp
    SUBTRACT: RateOp
    MULTIPLY: RateOp
    DIVIDE: RateOp
    POWER: RateOp
    MINIMUM: RateOp
    MAXIMUM: RateOp
    NEGATE: RateOp
    EXPONENTIAL: RateOp
    LOGARITHM: RateOp
    LESS: RateOp
    LESS_EQUAL: RateOp
    GREATER: RateOp
    GREATER_EQUAL: RateOp
    EQUAL: RateOp
    SELECT: RateOp
    SIGNAL: RateOp

class ConstraintRegion(Enum):
    OUTSIDE: ConstraintRegion
    INSIDE: ConstraintRegion

SphereRegion = ConstraintRegion

class ExternalConstraintKind(Enum):
    PLANE: ExternalConstraintKind
    SPHERE: ExternalConstraintKind
    BOX: ExternalConstraintKind
    CYLINDER: ExternalConstraintKind

class RodContactLocation(Enum):
    NEGATIVE: RodContactLocation
    POSITIVE: RodContactLocation
    INTERIOR: RodContactLocation

RodEndpoint = RodContactLocation

class SolverStatus(Enum):
    CONVERGED: SolverStatus
    ITERATION_LIMIT: SolverStatus
    BREAKDOWN: SolverStatus

class SolverBreakdown(Enum):
    NONE: SolverBreakdown
    NON_FINITE_RESIDUAL: SolverBreakdown
    NON_FINITE_CURVATURE: SolverBreakdown
    NON_POSITIVE_CURVATURE: SolverBreakdown

def backend_available(backend: BackendKind, device_index: int = 0) -> bool: ...
def backend_device_count(backend: BackendKind) -> int: ...

class Vec3:
    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None: ...

class BackendInfo:
    @property
    def kind(self) -> BackendKind: ...
    @property
    def name(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def device_index(self) -> int: ...
    @property
    def native(self) -> bool: ...

class GridBoundary:
    kind: GridBoundaryKind
    values: list[float]

    def __init__(self) -> None: ...
    def validate(self, signal_count: int) -> None: ...

class GridShape:
    x: int
    y: int
    z: int

    def __init__(self) -> None: ...

class SignalSolveParameters:
    max_iterations: int
    absolute_tolerance: float
    relative_tolerance: float

    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class SignalSolveReport:
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def residual_rms(self) -> float: ...

class SignalGridAffineReaction:
    source_rates: list[float]
    loss_rates: list[float]

    def __init__(self) -> None: ...
    def validate(self, level_count: int) -> None: ...

class SignalGridVelocityField:
    x_faces: list[float]
    y_faces: list[float]
    z_faces: list[float]

    def __init__(self) -> None: ...

class DepthAveragedFlowParameters:
    mean_inlet_speed: float
    axis: FlowAxis
    relative_tolerance: float
    max_iterations: int

    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class DepthAveragedFlowReport:
    @property
    def iterations(self) -> int: ...
    @property
    def residual(self) -> float: ...
    @property
    def relative_residual(self) -> float: ...
    @property
    def mean_inlet_speed(self) -> float: ...
    @property
    def max_speed(self) -> float: ...

class DepthAveragedFlowResult:
    @property
    def field(self) -> SignalGridVelocityField: ...
    @property
    def report(self) -> DepthAveragedFlowReport: ...

class ResolvedFlowParameters:
    mean_inlet_speed: float
    axis: FlowAxis
    relative_tolerance: float
    max_outer_iterations: int
    inner_relative_tolerance: float
    max_inner_iterations: int

    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class ResolvedFlowReport:
    @property
    def outer_iterations(self) -> int: ...
    @property
    def inner_iterations(self) -> int: ...
    @property
    def relative_residual(self) -> float: ...
    @property
    def momentum_relative_residual(self) -> float: ...
    @property
    def divergence_rms(self) -> float: ...
    @property
    def mean_inlet_speed(self) -> float: ...
    @property
    def max_speed(self) -> float: ...
    @property
    def min_gap_voxels(self) -> int: ...

class ResolvedFlowResult:
    @property
    def field(self) -> SignalGridVelocityField: ...
    @property
    def report(self) -> ResolvedFlowReport: ...

class SignalGridSpec:
    signal_count: int
    shape: GridShape
    origin: Vec3
    spacing: Vec3
    diffusion: list[float]
    advection: list[Vec3]
    reaction: SignalGridAffineReaction | None
    obstacles: list[int]
    velocity_field: SignalGridVelocityField | None
    integration: SignalIntegrationKind
    solver: SignalSolveParameters
    x_lower: GridBoundary
    x_upper: GridBoundary
    y_lower: GridBoundary
    y_upper: GridBoundary
    z_lower: GridBoundary
    z_upper: GridBoundary

    def __init__(self) -> None: ...
    @property
    def site_count(self) -> int: ...
    @property
    def level_count(self) -> int: ...
    @property
    def voxel_volume(self) -> float: ...
    def validate(self) -> None: ...

class _SignalGridCheckpoint:
    spec: SignalGridSpec
    levels: list[float]

    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class CellInit:
    position: Vec3
    direction: Vec3
    length: float
    radius: float
    growth_rate: float
    cell_type: int
    fixed: bool
    species: list[float]

    def __init__(self) -> None: ...

class CellSnapshot:
    id: int
    slot: int
    position: Vec3
    direction: Vec3
    length: float
    radius: float
    growth_rate: float
    cell_type: int
    fixed: bool
    species: list[float]

    def __init__(self) -> None: ...

class _LineageEntry:
    child: int
    parent: int

    def __init__(self) -> None: ...

class _WorldStateCheckpoint:
    species_count: int
    next_id: int
    cells: list[CellSnapshot]
    lineage: list[_LineageEntry]

    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class RateInstruction:
    operation: RateOp
    first: int
    second: int
    third: int
    value: float

    def __init__(self) -> None: ...

class SpeciesRatePlan:
    def __init__(
        self,
        species_count: int,
        instructions: list[RateInstruction],
        outputs: list[int],
    ) -> None: ...
    @staticmethod
    def zero(species_count: int) -> SpeciesRatePlan: ...
    @property
    def species_count(self) -> int: ...
    @property
    def instructions(self) -> list[RateInstruction]: ...
    @property
    def outputs(self) -> list[int]: ...
    def validate(self) -> None: ...

class CoupledRatePlan:
    def __init__(
        self,
        species_count: int,
        signal_count: int,
        instructions: list[RateInstruction],
        species_outputs: list[int],
        signal_outputs: list[int],
    ) -> None: ...
    @property
    def species_count(self) -> int: ...
    @property
    def signal_count(self) -> int: ...
    @property
    def instructions(self) -> list[RateInstruction]: ...
    @property
    def species_outputs(self) -> list[int]: ...
    @property
    def signal_outputs(self) -> list[int]: ...
    def validate(self) -> None: ...

class ContactParameters:
    activation_margin: float
    parallel_sine_threshold: float
    degeneracy_epsilon: float

    def __init__(self) -> None: ...

class CellContact:
    @property
    def first_id(self) -> int: ...
    @property
    def second_id(self) -> int: ...
    @property
    def first_slot(self) -> int: ...
    @property
    def second_slot(self) -> int: ...
    @property
    def ordinal(self) -> int: ...
    @property
    def point_on_first(self) -> Vec3: ...
    @property
    def normal(self) -> Vec3: ...
    @property
    def signed_separation(self) -> float: ...
    @property
    def weight(self) -> float: ...

class ContactGraph:
    @property
    def cell_count(self) -> int: ...
    @property
    def empty(self) -> bool: ...
    @property
    def contacts(self) -> list[CellContact]: ...
    def __len__(self) -> int: ...
    def incident_contact_indices(self, slot: int) -> list[int]: ...
    def neighbor_ids(self, slot: int) -> list[int]: ...

class PlaneConstraintInit:
    point: Vec3
    inward_normal: Vec3
    coefficient: float

    def __init__(self) -> None: ...

class SphereConstraintInit:
    center: Vec3
    radius: float
    coefficient: float
    allowed_region: SphereRegion

    def __init__(self) -> None: ...

class BoxConstraintInit:
    center: Vec3
    half_extents: Vec3
    coefficient: float
    allowed_region: ConstraintRegion

    def __init__(self) -> None: ...

class CylinderConstraintInit:
    center: Vec3
    radius: float
    half_height: float
    coefficient: float
    allowed_region: ConstraintRegion

    def __init__(self) -> None: ...

class _PlaneConstraint:
    id: int
    point: Vec3
    inward_normal: Vec3
    coefficient: float

    def __init__(self) -> None: ...

class _SphereConstraint:
    id: int
    center: Vec3
    radius: float
    coefficient: float
    allowed_region: SphereRegion

    def __init__(self) -> None: ...

class _BoxConstraint:
    id: int
    center: Vec3
    half_extents: Vec3
    coefficient: float
    allowed_region: ConstraintRegion

    def __init__(self) -> None: ...

class _CylinderConstraint:
    id: int
    center: Vec3
    radius: float
    half_height: float
    coefficient: float
    allowed_region: ConstraintRegion

    def __init__(self) -> None: ...

class _ConstraintSetCheckpoint:
    next_id: int
    planes: list[_PlaneConstraint]
    spheres: list[_SphereConstraint]
    boxes: list[_BoxConstraint]
    cylinders: list[_CylinderConstraint]

    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class _SimulationCheckpoint:
    schema_version: int
    time: float
    world: _WorldStateCheckpoint
    constraints: _ConstraintSetCheckpoint
    species_rate_plan: SpeciesRatePlan
    signal_grid: _SignalGridCheckpoint | None
    coupled_rate_plan: CoupledRatePlan | None
    culture: CultureCheckpoint | None

    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class ConstraintContactParameters:
    activation_margin: float
    degeneracy_epsilon: float

    def __init__(self) -> None: ...

class ExternalContact:
    @property
    def cell_id(self) -> int: ...
    @property
    def cell_slot(self) -> int: ...
    @property
    def constraint_id(self) -> int: ...
    @property
    def constraint_kind(self) -> ExternalConstraintKind: ...
    @property
    def location(self) -> RodContactLocation: ...
    @property
    def endpoint(self) -> RodContactLocation: ...
    @property
    def point_on_cell(self) -> Vec3: ...
    @property
    def normal(self) -> Vec3: ...
    @property
    def signed_separation(self) -> float: ...
    @property
    def weight(self) -> float: ...

class ExternalContactGraph:
    @property
    def cell_count(self) -> int: ...
    @property
    def empty(self) -> bool: ...
    @property
    def contacts(self) -> list[ExternalContact]: ...
    def __len__(self) -> int: ...
    def incident_contact_indices(self, slot: int) -> list[int]: ...

class CellCorrection:
    @property
    def translation(self) -> Vec3: ...
    @property
    def rotation(self) -> Vec3: ...
    @property
    def length(self) -> float: ...

class MechanicsParameters:
    mu_a: float
    gamma: float
    residual_rms_tolerance: float
    max_iterations: int

    def __init__(self) -> None: ...

class MechanicsIntegrationParameters:
    max_rotation_radians: float
    require_convergence: bool

    def __init__(self) -> None: ...

class SolverReport:
    @property
    def status(self) -> SolverStatus: ...
    @property
    def breakdown(self) -> SolverBreakdown: ...
    @property
    def iterations(self) -> int: ...
    @property
    def initial_residual_rms(self) -> float: ...
    @property
    def final_residual_rms(self) -> float: ...

class MechanicsSolveResult:
    @property
    def corrections(self) -> list[CellCorrection]: ...
    @property
    def report(self) -> SolverReport: ...

class Simulation:
    @overload
    def __init__(
        self,
        backend: BackendKind = BackendKind.CPU,
        reserved_capacity: int = 0,
        species_count: int = 0,
        device_index: int = 0,
    ) -> None: ...
    @overload
    def __init__(
        self,
        backend: BackendKind,
        checkpoint: _SimulationCheckpoint,
        device_index: int = 0,
    ) -> None: ...
    @property
    def backend_info(self) -> BackendInfo: ...
    def supports(self, feature: BackendFeature) -> bool: ...
    @property
    def time(self) -> float: ...
    @property
    def cell_count(self) -> int: ...
    @property
    def species_count(self) -> int: ...
    @property
    def signal_count(self) -> int: ...
    @property
    def has_signal_grid(self) -> bool: ...
    @property
    def last_signal_solve_report(self) -> SignalSolveReport | None: ...
    @property
    def has_coupled_rate_plan(self) -> bool: ...
    @property
    def has_culture(self) -> bool: ...
    @property
    def culture_checkpoint(self) -> CultureCheckpoint | None: ...
    @property
    def fluid_fragments(self) -> list[FluidFragment]: ...
    def configure_culture(
        self,
        *,
        fluid: StokesFlow,
        transport: SoluteTransport,
        cell_growth: Mapping[int, CellGrowth] = ...,
    ) -> None: ...
    @property
    def culture_state(self) -> CultureState | None: ...
    def _configure_culture(
        self,
        configuration: CultureConfiguration,
        concentrations: list[float] = ...,
        biochemical_volumes: list[float] = ...,
    ) -> None: ...
    def set_cell_force(
        self,
        id: int,
        force_n: tuple[float, float, float],
        torque_nm: tuple[float, float, float] = ...,
    ) -> None: ...
    def _restore_checkpoint(self, checkpoint: _SimulationCheckpoint) -> None: ...
    def cell_surface_concentrations(self, id: int) -> list[float]: ...
    def add_cell(self, cell: CellInit) -> int: ...
    def remove_cell(self, id: int) -> None: ...
    def apply_flow_drift(
        self, dt: float, integration: MechanicsIntegrationParameters = ...
    ) -> None: ...
    def add_plane_constraint(self, plane: PlaneConstraintInit) -> int: ...
    def add_sphere_constraint(self, sphere: SphereConstraintInit) -> int: ...
    def add_box_constraint(self, box: BoxConstraintInit) -> int: ...
    def add_cylinder_constraint(self, cylinder: CylinderConstraintInit) -> int: ...
    def set_cell_geometry(
        self, id: int, position: Vec3, direction: Vec3, length: float
    ) -> None: ...
    def set_cell_attributes(self, id: int, growth_rate: float, cell_type: int) -> None: ...
    def set_cell_fixed(self, id: int, fixed: bool) -> None: ...
    def set_species(self, id: int, levels: list[float]) -> None: ...
    def set_species_rate_plan(self, plan: SpeciesRatePlan) -> None: ...
    def set_coupled_rate_plan(self, plan: CoupledRatePlan) -> None: ...
    def clear_coupled_rate_plan(self) -> None: ...
    def configure_signal_grid(self, spec: SignalGridSpec, levels: list[float] = ...) -> None: ...
    def set_signal_levels(self, levels: list[float]) -> None: ...
    def set_velocity_field(self, field: SignalGridVelocityField | None) -> None: ...
    def set_signal_reaction(self, reaction: SignalGridAffineReaction | None) -> None: ...
    def divide(self, parent_id: int, first_fraction: float) -> tuple[int, int]: ...
    def divide_equal(self, parent_id: int) -> tuple[int, int]: ...
    def step(self, dt: float) -> None: ...
    def find_cell_contacts(self, parameters: ContactParameters = ...) -> ContactGraph: ...
    def find_external_contacts(
        self, parameters: ConstraintContactParameters = ...
    ) -> ExternalContactGraph: ...
    def solve_cell_mechanics(
        self,
        mechanics_parameters: MechanicsParameters = ...,
        contact_parameters: ContactParameters = ...,
        constraint_parameters: ConstraintContactParameters = ...,
    ) -> MechanicsSolveResult: ...
    def relax_cell_mechanics(
        self,
        mechanics_parameters: MechanicsParameters = ...,
        contact_parameters: ContactParameters = ...,
        integration_parameters: MechanicsIntegrationParameters = ...,
        constraint_parameters: ConstraintContactParameters = ...,
    ) -> MechanicsSolveResult: ...
    def solve_depth_averaged_flow(
        self,
        spec: SignalGridSpec,
        mobility: list[float] = ...,
        parameters: DepthAveragedFlowParameters = ...,
    ) -> DepthAveragedFlowResult: ...
    def solve_resolved_flow(
        self,
        spec: SignalGridSpec,
        drag: list[float] = ...,
        parameters: ResolvedFlowParameters = ...,
    ) -> ResolvedFlowResult: ...
    def cell(self, id: int) -> CellSnapshot: ...
    def cells(self) -> list[CellSnapshot]: ...
    def lineage_parent(self, id: int) -> int | None: ...
    @property
    def signal_levels(self) -> list[float]: ...
    def sample_signals(self, position: Vec3) -> list[float]: ...
    def _checkpoint(self) -> _SimulationCheckpoint: ...
    def validate(self) -> None: ...

class CapsuleBody:
    length_rate: float
    id: int
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    length: float
    radius: float
    fixed: bool
    force_n: tuple[float, float, float]
    torque_nm: tuple[float, float, float]
    def __init__(self) -> None: ...
    def validate(self) -> None: ...
    @property
    def geometric_volume(self) -> float: ...

class FluidBodyResult:
    @property
    def id(self) -> int: ...
    @property
    def velocity(self) -> tuple[float, float, float]: ...
    @property
    def angular_velocity(self) -> tuple[float, float, float]: ...
    @property
    def hydrodynamic_force_n(self) -> tuple[float, float, float]: ...
    @property
    def hydrodynamic_torque_nm(self) -> tuple[float, float, float]: ...
    @property
    def no_slip_rms_m_s(self) -> float: ...
    @property
    def volume_change_rate_m3_s(self) -> float: ...
    @property
    def marker_count(self) -> int: ...

class FluidBodyStepParameters:
    minimum_gap_m: float
    maximum_displacement_fraction: float
    max_halvings: int
    max_contact_iterations: int
    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class FluidContactResult:
    @property
    def first_id(self) -> int: ...
    @property
    def second_id(self) -> int: ...
    @property
    def normal(self) -> tuple[float, float, float]: ...
    @property
    def point_on_first(self) -> tuple[float, float, float]: ...
    @property
    def initial_gap_m(self) -> float: ...
    @property
    def normal_force_n(self) -> float: ...

class FluidBodyStep:
    @property
    def accepted_dt(self) -> float: ...
    @property
    def bodies(self) -> list[CapsuleBody]: ...
    @property
    def flow(self) -> FluidFlowResult: ...
    @property
    def contacts(self) -> list[FluidContactResult]: ...
    @property
    def halvings(self) -> int: ...
    @property
    def contact_iterations(self) -> int: ...

class FluidGridSpec:
    shape: GridShape
    origin: Vec3
    spacing: float
    length_unit_m: float
    time_unit_s: float
    obstacles: list[int]
    def __init__(self) -> None: ...
    def validate(self) -> None: ...
    @property
    def site_count(self) -> int: ...

class FluidProperties:
    viscosity_pa_s: float
    density_kg_m3: float
    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class FlowPortKind(Enum):
    PRESSURE: FlowPortKind
    FLOW_RATE: FlowPortKind

class FlowPort:
    name: str
    axis: FlowAxis
    upper: bool
    kind: FlowPortKind
    value: float
    sites: list[int]
    def __init__(self) -> None: ...

class LinearSolveParameters:
    relative_tolerance: float
    absolute_tolerance: float
    max_iterations: int
    memory_limit_bytes: int
    def __init__(self) -> None: ...
    def validate(self) -> None: ...

class FlowPortResult:
    @property
    def name(self) -> str: ...
    @property
    def pressure_pa(self) -> float: ...
    @property
    def flow_rate_m3_s(self) -> float: ...
    @property
    def area_m2(self) -> float: ...

class FluidSolveReport:
    def __init__(self) -> None: ...
    iterations: int
    relative_residual: float
    absolute_residual: float
    divergence_rms_per_s: float
    continuity_rms_per_s: float
    source_volume_rate_m3_s: float
    max_speed_m_s: float
    reynolds_number: float
    viscous_relaxation_time_s: float
    net_flow_rate_m3_s: float
    estimated_memory_bytes: int

class FluidFlowResult:
    @property
    def field(self) -> SignalGridVelocityField: ...
    @property
    def pressure_pa(self) -> list[float]: ...
    @property
    def ports(self) -> list[FlowPortResult]: ...
    @property
    def bodies(self) -> list[FluidBodyResult]: ...
    @property
    def report(self) -> FluidSolveReport: ...

class StokesFlowSolver:
    def __init__(self, backend: BackendKind = ..., device_index: int = 0) -> None: ...
    def solve(
        self,
        grid: FluidGridSpec,
        fluid: FluidProperties,
        ports: list[FlowPort],
        parameters: LinearSolveParameters = ...,
    ) -> FluidFlowResult: ...
    def solve_bodies(
        self,
        grid: FluidGridSpec,
        fluid: FluidProperties,
        ports: list[FlowPort],
        bodies: list[CapsuleBody],
        parameters: LinearSolveParameters = ...,
    ) -> FluidFlowResult: ...
    def propose_body_step(
        self,
        grid: FluidGridSpec,
        fluid: FluidProperties,
        ports: list[FlowPort],
        bodies: list[CapsuleBody],
        maximum_dt: float,
        solve_parameters: LinearSolveParameters = ...,
        step_parameters: FluidBodyStepParameters = ...,
    ) -> FluidBodyStep: ...

class FluidGeometryParameters:
    def __init__(self) -> None: ...
    surface_resolution: int
    maximum_surface_error_fraction: float
    memory_limit_bytes: int
    def validate(self) -> None: ...

class FluidFragment:
    @property
    def site(self) -> int: ...
    @property
    def component(self) -> int: ...
    @property
    def volume(self) -> float: ...
    @property
    def centroid(self) -> tuple[float, float, float]: ...

class FluidFace:
    @property
    def first(self) -> int: ...
    @property
    def second(self) -> int: ...
    @property
    def area(self) -> float: ...
    @property
    def centroid(self) -> tuple[float, float, float]: ...
    @property
    def normal(self) -> tuple[float, float, float]: ...
    @property
    def body_id(self) -> int: ...
    @property
    def axis(self) -> FlowAxis: ...
    @property
    def grid_face(self) -> int: ...

class FluidGeometryReport:
    @property
    def fluid_volume(self) -> float: ...
    @property
    def expected_fluid_volume(self) -> float: ...
    @property
    def volume_error(self) -> float: ...
    @property
    def maximum_surface_error(self) -> float: ...
    @property
    def component_count(self) -> int: ...
    @property
    def estimated_memory_bytes(self) -> int: ...

class FluidOverlap:
    @property
    def first(self) -> int: ...
    @property
    def second(self) -> int: ...
    @property
    def volume(self) -> float: ...

class Solute:
    amount_unit: str

    def __init__(self) -> None: ...
    name: str
    diffusion: float

class ChemicalBoundary:
    kind: ChemicalBoundaryKind
    allow_backflow: bool

    def __init__(self) -> None: ...
    port: str
    concentrations: list[float]

class SurfaceTransferLaw:
    def __init__(self) -> None: ...
    body_id: int
    solute: int
    uptake_velocity: float
    secretion_rate: float

class ChemicalTransfer:
    def __init__(self) -> None: ...
    port: str
    body_id: int
    amounts: list[float]

class SoluteTransportReport:
    def __init__(self) -> None: ...
    projection_iterations: int
    transport_iterations: int
    maximum_volume_residual: float
    mass_balance_error: list[float]

class SoluteTransportResult:
    @property
    def surfaces(self) -> list[SurfaceEnvironment]: ...
    @property
    def amounts(self) -> list[float]: ...
    @property
    def concentrations(self) -> list[float]: ...
    @property
    def reservoirs(self) -> list[ChemicalTransfer]: ...
    @property
    def cells(self) -> list[ChemicalTransfer]: ...
    @property
    def report(self) -> SoluteTransportReport: ...

class CellSurfaceExchange:
    def __init__(self) -> None: ...
    body_id: int
    solute: int
    species: int
    uptake_velocity: float
    secretion_rate: float

class ReserveRequirement:
    def __init__(self) -> None: ...
    species: int
    amount_per_biomass: float

class CultureConfiguration:
    growth: list[CellGrowthModel]
    events: list[CultureEvent]
    authoring_json: str
    coupling_tolerance: float
    maximum_coupling_iterations: int

    def __init__(self) -> None: ...
    grid: FluidGridSpec
    fluid: FluidProperties
    ports: list[FlowPort]
    solutes: list[Solute]
    reservoirs: list[ChemicalBoundary]
    exchange: list[CellSurfaceExchange]
    biomass_requirements: list[ReserveRequirement]
    biomass_per_geometric_volume: float
    solver: LinearSolveParameters
    stepping: FluidBodyStepParameters
    geometry: FluidGeometryParameters
    maximum_substeps: int
    maximum_retries: int
    def validate(self, species_count: int) -> None: ...

class CultureCellState:
    uptake_totals: list[float]
    realized_specific_rate: float
    biomass_produced: float

    def __init__(self) -> None: ...
    body: CapsuleBody
    biochemical_volume: float
    species_amounts: list[float]

class CultureReport:
    def __init__(self) -> None: ...
    substeps: int
    retries: int
    flow: FluidSolveReport
    transport: SoluteTransportReport

class CultureCheckpoint:
    time: float
    event_index: int

    def __init__(self) -> None: ...
    configuration: CultureConfiguration
    cells: list[CultureCellState]
    extracellular_amounts: list[float]
    reservoir_totals: list[ChemicalTransfer]
    last_report: CultureReport
    def validate(self, world: _WorldStateCheckpoint) -> None: ...

class FluidGeometry:
    def __init__(
        self,
        grid: FluidGridSpec,
        bodies: list[CapsuleBody],
        parameters: FluidGeometryParameters = ...,
    ) -> None: ...
    @property
    def grid(self) -> FluidGridSpec: ...
    @property
    def bodies(self) -> list[CapsuleBody]: ...
    @property
    def fragments(self) -> list[FluidFragment]: ...
    @property
    def faces(self) -> list[FluidFace]: ...
    @property
    def report(self) -> FluidGeometryReport: ...
    def overlaps(self, other: FluidGeometry) -> list[FluidOverlap]: ...

class SoluteTransportSolver:
    def propose(
        self,
        geometry: TransportGeometry,
        solutes: list[Solute],
        boundaries: list[ChemicalBoundary],
        amounts: list[float],
        exchange: list[SurfaceTransferLaw] = ...,
        parameters: LinearSolveParameters = ...,
    ) -> SoluteTransportResult: ...
    def __init__(self, backend: BackendKind = ..., device_index: int = 0) -> None: ...
    def step(
        self,
        before: FluidGeometry,
        after: FluidGeometry,
        velocity: SignalGridVelocityField,
        ports: list[FlowPort],
        solutes: list[Solute],
        reservoirs: list[ChemicalBoundary],
        amounts: list[float],
        dt: float,
        exchange: list[SurfaceTransferLaw] = ...,
        parameters: LinearSolveParameters = ...,
    ) -> SoluteTransportResult: ...

class ChemicalBoundaryKind(Enum):
    RESERVOIR_CONTACT = 0
    ADVECTIVE = 1
    OUTFLOW = 2

class GrowthKind(Enum):
    MONOD = 0
    ESSENTIAL = 1

class GrowthRequirement:
    def __init__(self) -> None: ...
    solute: int
    half_saturation: float
    biomass_yield: float

class CellGrowthModel:
    def __init__(self) -> None: ...
    cell_id: int
    kind: GrowthKind
    mu_max: float
    biomass_density: float
    volume_ratio: float
    requirements: list[GrowthRequirement]

class CultureEvent:
    def __init__(self) -> None: ...
    time: float
    ports: list[FlowPort]
    reservoirs: list[ChemicalBoundary]

MacVelocityField = SignalGridVelocityField

class GeometricFluxReport:
    @property
    def projection_iterations(self) -> int: ...
    @property
    def maximum_volume_residual(self) -> float: ...

class TransportGeometry:
    def __init__(
        self,
        before: FluidGeometry,
        after: FluidGeometry,
        velocity: MacVelocityField,
        ports: list[FlowPort],
        dt: float,
        backend: BackendKind = ...,
        device_index: int = 0,
        parameters: LinearSolveParameters = ...,
    ) -> None: ...
    @property
    def report(self) -> GeometricFluxReport: ...

class SurfaceEnvironment:
    @property
    def body_id(self) -> int: ...
    @property
    def area(self) -> float: ...
    @property
    def concentrations(self) -> list[float]: ...

class GrowthInput:
    def __init__(self) -> None: ...
    biochemical_volume: float
    surface_area: float
    concentrations: list[float]
    uptake: list[float]

class GrowthEvaluation:
    @property
    def uptake_velocities(self) -> list[float]: ...
    @property
    def biomass_gain(self) -> float: ...
    @property
    def biochemical_volume_gain(self) -> float: ...
    @property
    def geometric_volume_gain(self) -> float: ...
    @property
    def specific_rate(self) -> float: ...
    @property
    def stoichiometric_residual(self) -> float: ...

class GrowthExecutor:
    def __init__(self, backend: BackendKind = ..., device_index: int = 0) -> None: ...
    def evaluate(
        self, models: list[CellGrowthModel], inputs: list[GrowthInput], dt: float
    ) -> list[GrowthEvaluation]: ...
