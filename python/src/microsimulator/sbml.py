"""Compile a bounded SBML Core subset into native species-rate plans."""

# ruff: noqa: N802 -- protocol members mirror libSBML's public API.

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    RateInstruction,
    RateOp,
    SpeciesRatePlan,
)
from .channels import ChannelMetadata

_FLOAT32_MAX = 3.4028234663852886e38
_UINT32_MAX = (1 << 32) - 1


class SBMLImportError(ValueError):
    """Raised when an SBML document is invalid or outside the supported subset."""


@dataclass(frozen=True, slots=True)
class SBMLRateModel:
    """Backend-neutral result of compiling an SBML model."""

    model_id: str
    model_name: str
    species_ids: tuple[str, ...]
    species_names: tuple[str, ...]
    initial_levels: tuple[float, ...]
    rate_plan: SpeciesRatePlan
    warnings: tuple[str, ...]

    @property
    def channel_metadata(self) -> ChannelMetadata:
        """Use nonempty SBML names, falling back to stable SBML identifiers."""

        return ChannelMetadata(
            species=tuple(
                name if name.strip() else identifier
                for name, identifier in zip(self.species_names, self.species_ids, strict=True)
            )
        )

    @property
    def species_count(self) -> int:
        return len(self.species_ids)


class _SBMLError(Protocol):
    def getSeverity(self) -> int: ...
    def getMessage(self) -> str: ...
    def getLine(self) -> int: ...
    def getColumn(self) -> int: ...


class _ASTNode(Protocol):
    def getType(self) -> int: ...
    def getNumChildren(self) -> int: ...
    def getChild(self, index: int) -> _ASTNode: ...
    def isInteger(self) -> bool: ...
    def getInteger(self) -> int: ...
    def isReal(self) -> bool: ...
    def getReal(self) -> float: ...
    def isRational(self) -> bool: ...
    def getNumerator(self) -> int: ...
    def getDenominator(self) -> int: ...
    def isName(self) -> bool: ...
    def getName(self) -> str: ...


class _Parameter(Protocol):
    def getId(self) -> str: ...
    def getConstant(self) -> bool: ...
    def isSetValue(self) -> bool: ...
    def getValue(self) -> float: ...


class _Compartment(Protocol):
    def getId(self) -> str: ...
    def getConstant(self) -> bool: ...
    def isSetSize(self) -> bool: ...
    def getSize(self) -> float: ...


class _Species(Protocol):
    def getId(self) -> str: ...
    def getName(self) -> str: ...
    def getCompartment(self) -> str: ...
    def getBoundaryCondition(self) -> bool: ...
    def getConstant(self) -> bool: ...
    def getHasOnlySubstanceUnits(self) -> bool: ...
    def isSetConversionFactor(self) -> bool: ...
    def isSetInitialConcentration(self) -> bool: ...
    def getInitialConcentration(self) -> float: ...
    def isSetInitialAmount(self) -> bool: ...
    def getInitialAmount(self) -> float: ...


class _SpeciesReference(Protocol):
    def getSpecies(self) -> str: ...
    def getStoichiometry(self) -> float: ...
    def isSetStoichiometryMath(self) -> bool: ...
    def getConstant(self) -> bool: ...


class _KineticLaw(Protocol):
    def isSetMath(self) -> bool: ...
    def getMath(self) -> _ASTNode: ...
    def getNumLocalParameters(self) -> int: ...
    def getLocalParameter(self, index: int) -> _Parameter: ...
    def getNumParameters(self) -> int: ...
    def getParameter(self, index: int) -> _Parameter: ...


class _Reaction(Protocol):
    def getId(self) -> str: ...
    def isSetKineticLaw(self) -> bool: ...
    def getKineticLaw(self) -> _KineticLaw: ...
    def getNumReactants(self) -> int: ...
    def getReactant(self, index: int) -> _SpeciesReference: ...
    def getNumProducts(self) -> int: ...
    def getProduct(self, index: int) -> _SpeciesReference: ...


class _Model(Protocol):
    def getId(self) -> str: ...
    def getName(self) -> str: ...
    def getNumCompartments(self) -> int: ...
    def getCompartment(self, index: int) -> _Compartment: ...
    def getNumSpecies(self) -> int: ...
    def getSpecies(self, index: int) -> _Species: ...
    def getNumParameters(self) -> int: ...
    def getParameter(self, index: int) -> _Parameter: ...
    def getNumReactions(self) -> int: ...
    def getReaction(self, index: int) -> _Reaction: ...
    def getNumRules(self) -> int: ...
    def getNumEvents(self) -> int: ...
    def getNumConstraints(self) -> int: ...
    def getNumInitialAssignments(self) -> int: ...
    def getNumFunctionDefinitions(self) -> int: ...
    def isSetConversionFactor(self) -> bool: ...


class _Document(Protocol):
    def getLevel(self) -> int: ...
    def getVersion(self) -> int: ...
    def getNumErrors(self) -> int: ...
    def getError(self, index: int) -> _SBMLError: ...
    def checkConsistency(self) -> int: ...
    def getModel(self) -> _Model | None: ...


class _LibSBML(Protocol):
    LIBSBML_SEV_ERROR: int
    AST_PLUS: int
    AST_MINUS: int
    AST_TIMES: int
    AST_DIVIDE: int
    AST_POWER: int
    AST_FUNCTION_POWER: int
    AST_FUNCTION_EXP: int
    AST_FUNCTION_LN: int
    AST_CONSTANT_E: int
    AST_CONSTANT_PI: int
    AST_NAME_TIME: int

    def readSBMLFromString(self, source: str) -> _Document: ...


def _libsbml() -> _LibSBML:
    try:
        return cast(_LibSBML, import_module("libsbml"))
    except ImportError as error:
        raise SBMLImportError(
            "SBML import requires the optional dependency; install microsimulator[sbml]"
        ) from error


def _finite_float32(value: float, path: str) -> float:
    result = float(value)
    if not math.isfinite(result) or abs(result) > _FLOAT32_MAX:
        raise SBMLImportError(f"{path}: expected a finite float32 value")
    return result


def _diagnostic(error: _SBMLError) -> str:
    location = ""
    if error.getLine() > 0:
        location = f"line {error.getLine()}"
        if error.getColumn() > 0:
            location += f", column {error.getColumn()}"
        location += ": "
    return f"{location}{error.getMessage().strip()}"


def _validate_document(document: _Document, libsbml: _LibSBML) -> tuple[str, ...]:
    document.checkConsistency()
    failures: list[str] = []
    warnings: list[str] = []
    for index in range(document.getNumErrors()):
        error = document.getError(index)
        message = _diagnostic(error)
        if error.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            failures.append(message)
        else:
            warnings.append(message)
    if failures:
        detail = "; ".join(failures[:8])
        if len(failures) > 8:
            detail += f"; and {len(failures) - 8} more errors"
        raise SBMLImportError(f"SBML document is invalid: {detail}")
    return tuple(warnings)


def _reject_unsupported_model_features(model: _Model) -> None:
    unsupported = {
        "rules": model.getNumRules(),
        "events": model.getNumEvents(),
        "constraints": model.getNumConstraints(),
        "initial assignments": model.getNumInitialAssignments(),
        "function definitions": model.getNumFunctionDefinitions(),
    }
    present = [name for name, count in unsupported.items() if count != 0]
    if present:
        raise SBMLImportError(f"unsupported SBML constructs: {', '.join(present)}")
    if model.isSetConversionFactor():
        raise SBMLImportError("model conversion factors are not supported")


def _unit_compartment(model: _Model) -> tuple[str, float]:
    if model.getNumCompartments() != 1:
        raise SBMLImportError("SBML import requires exactly one compartment")
    compartment = model.getCompartment(0)
    identifier = compartment.getId()
    if not identifier:
        raise SBMLImportError("compartment must have an identifier")
    if not compartment.getConstant():
        raise SBMLImportError(f"compartment {identifier!r} must be constant")
    if not compartment.isSetSize():
        raise SBMLImportError(f"compartment {identifier!r} must declare size 1")
    size = _finite_float32(compartment.getSize(), f"compartment {identifier!r} size")
    if size != 1.0:
        raise SBMLImportError(f"compartment {identifier!r} must have size 1")
    return identifier, size


def _parameters(model: _Model) -> dict[str, float]:
    values: dict[str, float] = {}
    for index in range(model.getNumParameters()):
        parameter = model.getParameter(index)
        identifier = parameter.getId()
        if not identifier:
            raise SBMLImportError(f"global parameter {index} must have an identifier")
        if not parameter.getConstant():
            raise SBMLImportError(f"global parameter {identifier!r} must be constant")
        if not parameter.isSetValue():
            raise SBMLImportError(f"global parameter {identifier!r} must declare a value")
        values[identifier] = _finite_float32(
            parameter.getValue(), f"global parameter {identifier!r}"
        )
    return values


def _local_parameters(kinetic_law: _KineticLaw, reaction_id: str) -> dict[str, float]:
    values: dict[str, float] = {}
    parameters: list[tuple[_Parameter, bool]] = [
        (kinetic_law.getLocalParameter(index), False)
        for index in range(kinetic_law.getNumLocalParameters())
    ]
    parameters.extend(
        (kinetic_law.getParameter(index), True) for index in range(kinetic_law.getNumParameters())
    )
    for parameter, check_constant in parameters:
        identifier = parameter.getId()
        if not identifier or identifier in values:
            if identifier in values:
                continue
            raise SBMLImportError(f"reaction {reaction_id!r} has an unnamed local parameter")
        if check_constant and not parameter.getConstant():
            raise SBMLImportError(
                f"reaction {reaction_id!r} parameter {identifier!r} must be constant"
            )
        if not parameter.isSetValue():
            raise SBMLImportError(
                f"reaction {reaction_id!r} parameter {identifier!r} must declare a value"
            )
        values[identifier] = _finite_float32(
            parameter.getValue(), f"reaction {reaction_id!r} parameter {identifier!r}"
        )
    return values


class _PlanBuilder:
    def __init__(
        self,
        libsbml: _LibSBML,
        species: dict[str, int],
        global_parameters: dict[str, float],
        compartment: tuple[str, float],
    ) -> None:
        self.libsbml = libsbml
        self.species = species
        self.global_parameters = global_parameters
        self.compartment = compartment
        self.instructions: list[RateInstruction] = []

    def emit(
        self,
        operation: RateOp,
        *,
        first: int = 0,
        second: int = 0,
        value: float = 0.0,
    ) -> int:
        if len(self.instructions) >= _UINT32_MAX:
            raise SBMLImportError("SBML rate plan exceeds the uint32 instruction space")
        instruction = RateInstruction()
        instruction.operation = operation
        instruction.first = first
        instruction.second = second
        instruction.value = value
        self.instructions.append(instruction)
        return len(self.instructions) - 1

    def constant(self, value: float, path: str) -> int:
        return self.emit(RateOp.CONSTANT, value=_finite_float32(value, path))

    def fold(self, operation: RateOp, operands: list[int], path: str) -> int:
        if not operands:
            raise SBMLImportError(f"{path}: expression has no operands")
        result = operands[0]
        for operand in operands[1:]:
            result = self.emit(operation, first=result, second=operand)
        return result

    def expression(
        self,
        node: _ASTNode,
        local_parameters: dict[str, float],
        path: str,
    ) -> int:
        node_type = node.getType()
        child_count = node.getNumChildren()
        children = [node.getChild(index) for index in range(child_count)]
        if node.isRational():
            denominator = node.getDenominator()
            if denominator == 0:
                raise SBMLImportError(f"{path}: rational literal has zero denominator")
            return self.constant(node.getNumerator() / denominator, path)
        if node.isInteger():
            return self.constant(node.getInteger(), path)
        if node.isReal():
            return self.constant(node.getReal(), path)
        if node_type == self.libsbml.AST_CONSTANT_E:
            return self.constant(math.e, path)
        if node_type == self.libsbml.AST_CONSTANT_PI:
            return self.constant(math.pi, path)
        if node_type == self.libsbml.AST_NAME_TIME:
            raise SBMLImportError(f"{path}: time-dependent kinetic laws are not supported")
        if node.isName():
            name = node.getName()
            if name in local_parameters:
                return self.constant(local_parameters[name], path)
            if name in self.global_parameters:
                return self.constant(self.global_parameters[name], path)
            if name in self.species:
                return self.emit(RateOp.SPECIES, first=self.species[name])
            if name == self.compartment[0]:
                return self.constant(self.compartment[1], path)
            raise SBMLImportError(f"{path}: unresolved identifier {name!r}")

        compiled = [
            self.expression(child, local_parameters, f"{path}.child[{index}]")
            for index, child in enumerate(children)
        ]
        if node_type == self.libsbml.AST_PLUS:
            return self.fold(RateOp.ADD, compiled, path)
        if node_type == self.libsbml.AST_TIMES:
            return self.fold(RateOp.MULTIPLY, compiled, path)
        if node_type == self.libsbml.AST_MINUS:
            if child_count == 1:
                return self.emit(RateOp.NEGATE, first=compiled[0])
            if child_count == 2:
                return self.emit(RateOp.SUBTRACT, first=compiled[0], second=compiled[1])
            raise SBMLImportError(f"{path}: subtraction requires one or two operands")
        if node_type == self.libsbml.AST_DIVIDE:
            if child_count != 2:
                raise SBMLImportError(f"{path}: division requires two operands")
            return self.emit(RateOp.DIVIDE, first=compiled[0], second=compiled[1])
        if node_type in {self.libsbml.AST_POWER, self.libsbml.AST_FUNCTION_POWER}:
            if child_count != 2:
                raise SBMLImportError(f"{path}: power requires two operands")
            return self.emit(RateOp.POWER, first=compiled[0], second=compiled[1])
        if node_type == self.libsbml.AST_FUNCTION_EXP:
            if child_count != 1:
                raise SBMLImportError(f"{path}: exponential requires one operand")
            return self.emit(RateOp.EXPONENTIAL, first=compiled[0])
        if node_type == self.libsbml.AST_FUNCTION_LN:
            if child_count != 1:
                raise SBMLImportError(f"{path}: natural logarithm requires one operand")
            return self.emit(RateOp.LOGARITHM, first=compiled[0])
        raise SBMLImportError(f"{path}: unsupported MathML node type {node_type}")


def _species_metadata(
    model: _Model, compartment_id: str
) -> tuple[list[_Species], tuple[str, ...], tuple[str, ...], tuple[float, ...]]:
    species_values: list[_Species] = []
    identifiers: list[str] = []
    names: list[str] = []
    levels: list[float] = []
    for index in range(model.getNumSpecies()):
        species = model.getSpecies(index)
        identifier = species.getId()
        if not identifier:
            raise SBMLImportError(f"species {index} must have an identifier")
        if identifier in identifiers:
            raise SBMLImportError(f"duplicate species identifier {identifier!r}")
        if species.getCompartment() != compartment_id:
            raise SBMLImportError(
                f"species {identifier!r} is not in compartment {compartment_id!r}"
            )
        if species.getHasOnlySubstanceUnits():
            raise SBMLImportError(f"species {identifier!r} must be concentration-valued")
        if species.isSetConversionFactor():
            raise SBMLImportError(f"species {identifier!r} conversion factors are not supported")
        has_concentration = species.isSetInitialConcentration()
        has_amount = species.isSetInitialAmount()
        if has_concentration == has_amount:
            raise SBMLImportError(
                f"species {identifier!r} must declare exactly one initial concentration or amount"
            )
        initial = (
            species.getInitialConcentration() if has_concentration else species.getInitialAmount()
        )
        initial = _finite_float32(initial, f"species {identifier!r} initial level")
        if initial < 0.0:
            raise SBMLImportError(f"species {identifier!r} initial level must be non-negative")
        species_values.append(species)
        identifiers.append(identifier)
        names.append(species.getName() or identifier)
        levels.append(initial)
    return species_values, tuple(identifiers), tuple(names), tuple(levels)


def _stoichiometry(reference: _SpeciesReference, reaction_id: str) -> float:
    if reference.isSetStoichiometryMath() or not reference.getConstant():
        raise SBMLImportError(f"reaction {reaction_id!r} uses dynamic stoichiometry")
    value = _finite_float32(reference.getStoichiometry(), f"reaction {reaction_id!r} stoichiometry")
    if value < 0.0:
        raise SBMLImportError(f"reaction {reaction_id!r} stoichiometry must be non-negative")
    return value


def _compile_model(model: _Model, libsbml: _LibSBML, warnings: tuple[str, ...]) -> SBMLRateModel:
    _reject_unsupported_model_features(model)
    compartment = _unit_compartment(model)
    species_values, species_ids, species_names, initial_levels = _species_metadata(
        model, compartment[0]
    )
    species_indices = {identifier: index for index, identifier in enumerate(species_ids)}
    builder = _PlanBuilder(libsbml, species_indices, _parameters(model), compartment)
    zero = builder.constant(0.0, "zero derivative") if species_values else 0
    contributions: list[list[tuple[float, int]]] = [[] for _ in species_values]

    for reaction_index in range(model.getNumReactions()):
        reaction = model.getReaction(reaction_index)
        reaction_id = reaction.getId() or f"reaction[{reaction_index}]"
        if not reaction.isSetKineticLaw():
            raise SBMLImportError(f"reaction {reaction_id!r} must declare a kinetic law")
        kinetic_law = reaction.getKineticLaw()
        if not kinetic_law.isSetMath():
            raise SBMLImportError(f"reaction {reaction_id!r} kinetic law must contain MathML")
        rate = builder.expression(
            kinetic_law.getMath(),
            _local_parameters(kinetic_law, reaction_id),
            f"reaction {reaction_id!r} kinetic law",
        )
        for sign, count, getter in (
            (-1.0, reaction.getNumReactants(), reaction.getReactant),
            (1.0, reaction.getNumProducts(), reaction.getProduct),
        ):
            for reference_index in range(count):
                reference = getter(reference_index)
                identifier = reference.getSpecies()
                if identifier not in species_indices:
                    raise SBMLImportError(
                        f"reaction {reaction_id!r} references unknown species {identifier!r}"
                    )
                coefficient = sign * _stoichiometry(reference, reaction_id)
                contributions[species_indices[identifier]].append((coefficient, rate))

    outputs: list[int] = []
    for index, species in enumerate(species_values):
        if species.getBoundaryCondition() or species.getConstant():
            outputs.append(zero)
            continue
        output = zero
        for coefficient, rate in contributions[index]:
            if coefficient == 1.0:
                output = builder.emit(RateOp.ADD, first=output, second=rate)
            elif coefficient == -1.0:
                output = builder.emit(RateOp.SUBTRACT, first=output, second=rate)
            else:
                factor = builder.constant(
                    coefficient, f"species {species_ids[index]!r} coefficient"
                )
                term = builder.emit(RateOp.MULTIPLY, first=factor, second=rate)
                output = builder.emit(RateOp.ADD, first=output, second=term)
        outputs.append(output)

    try:
        rate_plan = SpeciesRatePlan(len(species_values), builder.instructions, outputs)
    except (ValueError, OverflowError) as error:
        raise SBMLImportError(f"compiled SBML rate plan is invalid: {error}") from error
    return SBMLRateModel(
        model_id=model.getId(),
        model_name=model.getName(),
        species_ids=species_ids,
        species_names=species_names,
        initial_levels=initial_levels,
        rate_plan=rate_plan,
        warnings=warnings,
    )


def parse_sbml(source: str) -> SBMLRateModel:
    """Parse SBML XML and compile the supported subset to a native rate plan."""

    if not source.strip():
        raise SBMLImportError("SBML source must be a nonempty string")
    libsbml = _libsbml()
    document = libsbml.readSBMLFromString(source)
    if document.getLevel() > 0 and (document.getLevel() != 3 or document.getVersion() != 2):
        raise SBMLImportError("SBML import currently requires Level 3 Version 2 Core")
    warnings = _validate_document(document, libsbml)
    model = document.getModel()
    if model is None:
        raise SBMLImportError("SBML document does not contain a model")
    return _compile_model(model, libsbml, warnings)


def load_sbml(path: str | Path) -> SBMLRateModel:
    """Read an SBML file as UTF-8 and compile the supported subset."""

    source_path = Path(path)
    try:
        source = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SBMLImportError(f"could not read SBML file {source_path}") from error
    return parse_sbml(source)
