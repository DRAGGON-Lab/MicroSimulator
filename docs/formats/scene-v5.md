# Scene format version 5

Version 5 keeps the data-only scene envelope, SHA-256 frame digest, cell rendering geometry, grid projection, and channel metadata from [version 4](scene-v4.md). The optional `frame.culture` object contains the resolved fluid, chemical, and biological state. Its name reflects the coupled simulation; a biological medium recipe is not a fluid solver or a scene.

The culture object retains the physical units, fluid properties, grid geometry, solute identities, fragments, cells, cumulative boundary transfers, and diagnostics defined for the version-4 `media` object. It adds `solute_amount_units`, an ordered array matching `solutes`, with the native amount basis for each chemical.

Each culture cell additionally includes:

| Field                             | Meaning                                                               |
| --------------------------------- | --------------------------------------------------------------------- |
| `dry_biomass_g`                   | Derived dry mass in grams for a kinetic growth model, otherwise null  |
| `realized_specific_rate_per_hour` | Accepted specific biochemical-volume growth over the last public step |
| `biomass_produced_g`              | Cumulative biomass produced by accepted nutrient uptake               |
| `uptake_totals`                   | Cumulative consumed amounts in the declared solute order and units    |

The biochemical volume, intracellular amounts, fragment volumes, and extracellular amounts retain their existing meanings. Fluid-volume-weighted voxel concentrations are a display projection; conservation calculations use fragment amounts. Culture cells must match frame cells by identity and order. Solute and amount arrays must have matching dimensions, and nonfinite or negative amounts are rejected.

Writers emit version 5. Readers accept versions 2–5, authenticate the original payload, and validate its original closed schema before constructing the current scene representation. Version 4 uses `media` and receives unknown amount units (`model`), null dry mass, and zero growth/uptake diagnostics. Version 2 and 3 have no culture state. Renaming an in-memory field does not alter an archived frame or its digest.
