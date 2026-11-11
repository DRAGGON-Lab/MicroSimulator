from collections.abc import Mapping, Sequence
from enum import Enum
from os import PathLike
from typing import overload

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

class ChannelMetadataError(ValueError): ...
class CheckpointError(ValueError): ...
class SceneError(ValueError): ...

class ChannelMetadata:
    def __init__(
        self,
        species: tuple[str | None, ...] | None = None,
        signals: tuple[str | None, ...] | None = None,
    ) -> None: ...
    @property
    def species(self) -> tuple[str | None, ...] | None: ...
    @property
    def signals(self) -> tuple[str | None, ...] | None: ...
    def resolved(self, species_count: int, signal_count: int) -> ChannelMetadata: ...
    def to_json(self, species_count: int, signal_count: int) -> dict[str, list[str | None]]: ...
    @staticmethod
    def from_json(value: object, species_count: int, signal_count: int) -> ChannelMetadata: ...
    def replace(
        self,
        *,
        species: tuple[str | None, ...] | None = ...,
        signals: tuple[str | None, ...] | None = ...,
    ) -> ChannelMetadata: ...

CHECKPOINT_FORMAT: str
CHECKPOINT_VERSION: int
MAX_CHECKPOINT_BYTES: int

type _JSON = str | int | float | bool | list[_JSON] | dict[str, _JSON] | None

class _CheckpointDocument:
    @property
    def simulation(self) -> _SimulationCheckpoint: ...
    @property
    def controller(self) -> _JSON: ...
    @property
    def provenance(self) -> dict[str, _JSON]: ...
    @property
    def source_backend(self) -> BackendInfo: ...
    @property
    def channel_metadata(self) -> ChannelMetadata: ...

def _save_checkpoint(
    simulation: Simulation,
    path: str | PathLike[str],
    provenance: Mapping[str, _JSON],
    controller: _JSON,
    channel_metadata: ChannelMetadata,
) -> None: ...
def _load_checkpoint(path: str | PathLike[str]) -> _CheckpointDocument: ...

SCENE_FORMAT: str
SCENE_VERSION: int
MAX_SCENE_BYTES: int
MAX_SCENE_CHANNELS: int

class SceneBackend:
    def __init__(
        self, kind: str, name: str, device: str, device_index: int, native: bool
    ) -> None: ...
    @property
    def kind(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def device_index(self) -> int: ...
    @property
    def native(self) -> bool: ...
    def replace(
        self,
        *,
        kind: str = ...,
        name: str = ...,
        device: str = ...,
        device_index: int = ...,
        native: bool = ...,
    ) -> SceneBackend: ...

class SceneCell:
    def __init__(
        self,
        id: int,
        parent_id: int | None,
        slot: int,
        position: tuple[float, float, float],
        direction: tuple[float, float, float],
        length: float,
        radius: float,
        growth_rate: float,
        cell_type: int,
        fixed: bool,
        species: Sequence[float],
    ) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def parent_id(self) -> int | None: ...
    @property
    def slot(self) -> int: ...
    @property
    def position(self) -> tuple[float, float, float]: ...
    @property
    def direction(self) -> tuple[float, float, float]: ...
    @property
    def length(self) -> float: ...
    @property
    def radius(self) -> float: ...
    @property
    def growth_rate(self) -> float: ...
    @property
    def cell_type(self) -> int: ...
    @property
    def fixed(self) -> bool: ...
    @property
    def species(self) -> tuple[float, ...]: ...
    def replace(
        self,
        *,
        id: int = ...,
        parent_id: int | None = ...,
        slot: int = ...,
        position: tuple[float, float, float] = ...,
        direction: tuple[float, float, float] = ...,
        length: float = ...,
        radius: float = ...,
        growth_rate: float = ...,
        cell_type: int = ...,
        fixed: bool = ...,
        species: Sequence[float] = ...,
    ) -> SceneCell: ...

class SceneGridBoundary:
    def __init__(self, kind: str, values: Sequence[float]) -> None: ...
    @property
    def kind(self) -> str: ...
    @property
    def values(self) -> tuple[float, ...]: ...
    def replace(self, *, kind: str = ..., values: Sequence[float] = ...) -> SceneGridBoundary: ...

class SceneSignalGrid:
    def __init__(
        self,
        signal_count: int,
        shape: tuple[int, int, int],
        origin: tuple[float, float, float],
        spacing: tuple[float, float, float],
        x_lower: SceneGridBoundary,
        x_upper: SceneGridBoundary,
        y_lower: SceneGridBoundary,
        y_upper: SceneGridBoundary,
        z_lower: SceneGridBoundary,
        z_upper: SceneGridBoundary,
        levels: Sequence[float],
    ) -> None: ...
    @property
    def signal_count(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    @property
    def origin(self) -> tuple[float, float, float]: ...
    @property
    def spacing(self) -> tuple[float, float, float]: ...
    @property
    def x_lower(self) -> SceneGridBoundary: ...
    @property
    def x_upper(self) -> SceneGridBoundary: ...
    @property
    def y_lower(self) -> SceneGridBoundary: ...
    @property
    def y_upper(self) -> SceneGridBoundary: ...
    @property
    def z_lower(self) -> SceneGridBoundary: ...
    @property
    def z_upper(self) -> SceneGridBoundary: ...
    @property
    def levels(self) -> tuple[float, ...]: ...
    def replace(
        self,
        *,
        signal_count: int = ...,
        shape: tuple[int, int, int] = ...,
        origin: tuple[float, float, float] = ...,
        spacing: tuple[float, float, float] = ...,
        x_lower: SceneGridBoundary = ...,
        x_upper: SceneGridBoundary = ...,
        y_lower: SceneGridBoundary = ...,
        y_upper: SceneGridBoundary = ...,
        z_lower: SceneGridBoundary = ...,
        z_upper: SceneGridBoundary = ...,
        levels: Sequence[float] = ...,
    ) -> SceneSignalGrid: ...

class ScenePlaneConstraint:
    def __init__(
        self,
        id: int,
        point: tuple[float, float, float],
        inward_normal: tuple[float, float, float],
        coefficient: float,
    ) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def point(self) -> tuple[float, float, float]: ...
    @property
    def inward_normal(self) -> tuple[float, float, float]: ...
    @property
    def coefficient(self) -> float: ...
    def replace(
        self,
        *,
        id: int = ...,
        point: tuple[float, float, float] = ...,
        inward_normal: tuple[float, float, float] = ...,
        coefficient: float = ...,
    ) -> ScenePlaneConstraint: ...

class SceneSphereConstraint:
    def __init__(
        self,
        id: int,
        center: tuple[float, float, float],
        radius: float,
        coefficient: float,
        allowed_region: str,
    ) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def center(self) -> tuple[float, float, float]: ...
    @property
    def radius(self) -> float: ...
    @property
    def coefficient(self) -> float: ...
    @property
    def allowed_region(self) -> str: ...
    def replace(
        self,
        *,
        id: int = ...,
        center: tuple[float, float, float] = ...,
        radius: float = ...,
        coefficient: float = ...,
        allowed_region: str = ...,
    ) -> SceneSphereConstraint: ...

class SceneBoxConstraint:
    def __init__(
        self,
        id: int,
        center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        coefficient: float,
        allowed_region: str,
    ) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def center(self) -> tuple[float, float, float]: ...
    @property
    def half_extents(self) -> tuple[float, float, float]: ...
    @property
    def coefficient(self) -> float: ...
    @property
    def allowed_region(self) -> str: ...
    def replace(
        self,
        *,
        id: int = ...,
        center: tuple[float, float, float] = ...,
        half_extents: tuple[float, float, float] = ...,
        coefficient: float = ...,
        allowed_region: str = ...,
    ) -> SceneBoxConstraint: ...

class SceneCylinderConstraint:
    def __init__(
        self,
        id: int,
        center: tuple[float, float, float],
        radius: float,
        half_height: float,
        coefficient: float,
        allowed_region: str,
    ) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def center(self) -> tuple[float, float, float]: ...
    @property
    def radius(self) -> float: ...
    @property
    def half_height(self) -> float: ...
    @property
    def coefficient(self) -> float: ...
    @property
    def allowed_region(self) -> str: ...
    def replace(
        self,
        *,
        id: int = ...,
        center: tuple[float, float, float] = ...,
        radius: float = ...,
        half_height: float = ...,
        coefficient: float = ...,
        allowed_region: str = ...,
    ) -> SceneCylinderConstraint: ...

class SceneConstraints:
    def __init__(
        self,
        planes: Sequence[ScenePlaneConstraint],
        spheres: Sequence[SceneSphereConstraint],
        boxes: Sequence[SceneBoxConstraint],
        cylinders: Sequence[SceneCylinderConstraint],
    ) -> None: ...
    @property
    def planes(self) -> tuple[ScenePlaneConstraint, ...]: ...
    @property
    def spheres(self) -> tuple[SceneSphereConstraint, ...]: ...
    @property
    def boxes(self) -> tuple[SceneBoxConstraint, ...]: ...
    @property
    def cylinders(self) -> tuple[SceneCylinderConstraint, ...]: ...
    def replace(
        self,
        *,
        planes: Sequence[ScenePlaneConstraint] = ...,
        spheres: Sequence[SceneSphereConstraint] = ...,
        boxes: Sequence[SceneBoxConstraint] = ...,
        cylinders: Sequence[SceneCylinderConstraint] = ...,
    ) -> SceneConstraints: ...

class SceneFrame:
    def __init__(
        self,
        time: float,
        backend: SceneBackend,
        species_count: int,
        cells: Sequence[SceneCell],
        constraints: SceneConstraints,
        signal_grid: SceneSignalGrid | None,
        channel_metadata: ChannelMetadata = ...,
    ) -> None: ...
    @property
    def time(self) -> float: ...
    @property
    def backend(self) -> SceneBackend: ...
    @property
    def species_count(self) -> int: ...
    @property
    def cells(self) -> tuple[SceneCell, ...]: ...
    @property
    def constraints(self) -> SceneConstraints: ...
    @property
    def signal_grid(self) -> SceneSignalGrid | None: ...
    @property
    def channel_metadata(self) -> ChannelMetadata: ...
    def replace(
        self,
        *,
        time: float = ...,
        backend: SceneBackend = ...,
        species_count: int = ...,
        cells: Sequence[SceneCell] = ...,
        constraints: SceneConstraints = ...,
        signal_grid: SceneSignalGrid | None = ...,
        channel_metadata: ChannelMetadata = ...,
    ) -> SceneFrame: ...

def capture_scene(
    simulation: Simulation, *, channel_metadata: ChannelMetadata = ...
) -> SceneFrame: ...
def dumps_scene(frame: SceneFrame) -> str: ...
def save_scene(frame: SceneFrame, path: str | PathLike[str]) -> None: ...
def parse_scene(source: str | bytes) -> SceneFrame: ...
def load_scene(path: str | PathLike[str]) -> SceneFrame: ...
