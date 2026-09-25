from __future__ import annotations

import math
from pathlib import Path

import pytest
from microsimulator import (
    BackendFeature,
    BackendKind,
    CellInit,
    SBMLImportError,
    Simulation,
    backend_available,
    load_sbml,
    parse_sbml,
)


def _model_xml(
    *,
    compartment_size: str = "1",
    kinetic_math: str = "<apply><times/><ci>k</ci><ci>A</ci></apply>",
    local_parameters: str = "",
    extra_model: str = "",
    product_boundary: str = "false",
    parameter_constant: str = "true",
    reactant_id: str = "",
) -> str:
    reactant_id_attribute = f' id="{reactant_id}"' if reactant_id else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="conversion" name="A to B">
    <listOfCompartments>
      <compartment id="cell" size="{compartment_size}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" name="substrate" compartment="cell" initialConcentration="3"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" name="product" compartment="cell" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="{product_boundary}"
               constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" constant="{parameter_constant}"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="conversion_reaction" reversible="false">
        <listOfReactants>
          <speciesReference{reactant_id_attribute} species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="2" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          {local_parameters}
          <math xmlns="http://www.w3.org/1998/Math/MathML">{kinetic_math}</math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    {extra_model}
  </model>
</sbml>
"""


def _run_model(backend: BackendKind, *, local_rate: bool = False) -> tuple[float, float]:
    local_parameters = (
        '<listOfLocalParameters><localParameter id="k" value="2"/></listOfLocalParameters>'
        if local_rate
        else ""
    )
    model = parse_sbml(_model_xml(local_parameters=local_parameters))
    simulation = Simulation(backend, species_count=model.species_count)
    cell = CellInit()
    cell.growth_rate = 0.0
    cell.species = list(model.initial_levels)
    cell_id = simulation.add_cell(cell)
    simulation.set_species_rate_plan(model.rate_plan)
    simulation.step(0.1)
    levels = simulation.cell(cell_id).species
    return levels[0], levels[1]


def test_sbml_compiles_to_native_rate_plan_on_available_backends() -> None:
    model = parse_sbml(_model_xml())
    assert model.model_id == "conversion"
    assert model.model_name == "A to B"
    assert model.species_ids == ("A", "B")
    assert model.species_names == ("substrate", "product")
    assert model.initial_levels == (3.0, 0.0)
    assert model.species_count == 2
    model.rate_plan.validate()

    for backend in BackendKind:
        if not backend_available(backend):
            continue
        probe = Simulation(backend)
        if not probe.supports(BackendFeature.SPECIES):
            continue
        substrate, product = _run_model(backend)
        assert math.isclose(substrate, 2.85, rel_tol=1.0e-6, abs_tol=1.0e-6)
        assert math.isclose(product, 0.3, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_local_parameter_shadows_global_parameter() -> None:
    substrate, product = _run_model(BackendKind.CPU, local_rate=True)
    assert math.isclose(substrate, 2.4, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert math.isclose(product, 1.2, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_arithmetic_power_exponential_and_logarithm_compile() -> None:
    kinetic_math = """
<apply>
  <plus/>
  <apply><times/><ci>k</ci><apply><power/><ci>A</ci><cn>2</cn></apply></apply>
  <apply><exp/><apply><ln/><ci>A</ci></apply></apply>
</apply>
"""
    model = parse_sbml(_model_xml(kinetic_math=kinetic_math))
    simulation = Simulation(species_count=2)
    cell = CellInit()
    cell.growth_rate = 0.0
    cell.species = list(model.initial_levels)
    cell_id = simulation.add_cell(cell)
    simulation.set_species_rate_plan(model.rate_plan)
    simulation.step(0.1)
    levels = simulation.cell(cell_id).species
    assert math.isclose(levels[0], 2.25, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert math.isclose(levels[1], 1.5, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_boundary_species_has_zero_derivative() -> None:
    model = parse_sbml(_model_xml(product_boundary="true"))
    simulation = Simulation(species_count=2)
    cell = CellInit()
    cell.growth_rate = 0.0
    cell.species = list(model.initial_levels)
    cell_id = simulation.add_cell(cell)
    simulation.set_species_rate_plan(model.rate_plan)
    simulation.step(0.1)
    assert simulation.cell(cell_id).species[1] == 0.0


def test_sbml_file_loader_matches_string_loader(tmp_path: Path) -> None:
    path = tmp_path / "conversion.xml"
    path.write_text(_model_xml(), encoding="utf-8")
    loaded = load_sbml(path)
    parsed = parse_sbml(_model_xml())
    assert loaded.species_ids == parsed.species_ids
    assert loaded.initial_levels == parsed.initial_levels
    assert loaded.rate_plan.outputs == parsed.rate_plan.outputs


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (_model_xml(compartment_size="2"), "size 1"),
        (
            _model_xml(
                parameter_constant="false",
                extra_model='<listOfRules><assignmentRule variable="k"><math '
                'xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>'
                "</assignmentRule></listOfRules>",
            ),
            "rules",
        ),
        (_model_xml(kinetic_math="<apply><sin/><ci>A</ci></apply>"), "unsupported MathML"),
        (
            _model_xml(kinetic_math="<ci>reactant_ref</ci>", reactant_id="reactant_ref"),
            "unresolved identifier",
        ),
    ],
)
def test_unsupported_sbml_semantics_fail_explicitly(source: str, message: str) -> None:
    with pytest.raises(SBMLImportError, match=message):
        parse_sbml(source)


def test_malformed_or_empty_sbml_fails_explicitly() -> None:
    with pytest.raises(SBMLImportError, match="nonempty"):
        parse_sbml("")
    with pytest.raises(SBMLImportError, match="invalid"):
        parse_sbml("<sbml>")

    level_two = _model_xml().replace('level="3" version="2"', 'level="2" version="5"').replace(
        "/level3/version2/core", "/level2/version5"
    )
    with pytest.raises(SBMLImportError, match="Level 3 Version 2"):
        parse_sbml(level_two)


def test_sbml_channel_metadata_uses_names_and_identifier_fallback() -> None:
    source = _model_xml().replace('name="substrate"', 'name=""')
    model = parse_sbml(source)
    assert model.channel_metadata.species == ("A", "product")
    assert model.channel_metadata.signals is None
