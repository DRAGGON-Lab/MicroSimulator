# Independent viewer review, round 1

This is an archived independent review. See the [round summary](../feedback-review.md) for final heads, corrections and combined validation. Temporary evidence paths in this report identify local artifacts; they are not repository dependencies.

Reviewed contracts, pinned diffs, production implementation, and original evidence maps for:

| Issue | PR | Reviewed SHA | Base |
| --- | --- | --- | --- |
| 20 | 27 | 2a6bdc0340994afb1c6fe8b6d9f92ecc11257c18 | master baseline b69193b9 |
| 15 | 30 | e96672c4e8e40a827e04a7ae95dcf9e3f27cca2d | 571254a967ffc6a545832fa8dc013e2f1d714d4f |
| 23 | 29 | c1cb4d6fd20d0f0ca5f5bb59f3164413e4ca1f97 | 571254a967ffc6a545832fa8dc013e2f1d714d4f |
| 19 | 32 | 9a27f8714de8515d6cb31a0b37ab5d1ce3bfa0c2 | marpaia/base-19 |
| 21 | 36 | 178143550dc11f63689f8eba5b412146224b7aea | master baseline b69193b9 |

Review completed: **no confirmed material findings** in PRs27,30,29,32,36. No source fixes requested. No worktree source changes or GitHub actions made. All five issue worktrees remained clean at completion.

## Source-grounded checks completed

- #20: `DatasetReferenceGrid` stores immutable initial extent/position/spacing and resets only in explicit beginDataset. Finite box/sphere/cylinder bounds precede colony fallback; planes do not influence reference extent. Ten-unit fallback covers initially empty scenes. `setFrame` defaults to preserving camera and `fitColony` no longer configures grid. Main presentation state retains requested channel/slice indices across temporarily missing data, returning clamped effective values without overwriting preferences. Committed tests cover growth, translation/Z, division/removal/reset/reversed time, new dataset, empty initial state, finite-device priority and plane exclusion. Browser test source checks actual matrices/projection, orbit/pan/zoom/Fit and file transitions.
- #15: Scalar math validates finite lower<upper, handles constant automatic midpoint and empty ranges, clips mapping input without mutating data, and avoids overflowing difference for extreme finite bounds. Per-kind/per-index ranges and saved fixed bounds reset only on dataset opening. Editor separates invalid/draft text from active config and uses native accessible form controls. Committed browser source covers actual instance RGB and signal texture bytes, shared value 2 across [2,4]/[2,40], endpoint mapping and inspector, invalid input, missing/returning channels, keyboard submission and new files.
- #23: One persistent device Group.visible toggles meshes and edge siblings; normal rebuilds do not reset group visibility. Main retains preference when control disabled due to absent geometry and explicitly restores enabled for new datasets. Setter has no camera, bounds, simulation or mask mutation.
- #19: Shared range storage is numerical-index keyed. CSS tints decode into linear RGB, each independently normalized contribution is added, final components clamp to1, result encodes to sRGB for existing renderer boundary. Canonical accumulation order provides reorder independence. Disabled channels are excluded; no channels uses documented neutral gray. Controls cache rows and state across missing channels, and active-config identity check synchronizes single/composite editors. Unit and browser test source cover primary colors, half-intensity color space, shared editor updates and moving-cell selection.
- #21: Open cylinders and complementary hemispheres have matching azimuth/sample rings and normals. Uniform cap scale preserves radius independently of centerline length; all parts share orientation, and zero-length rod collapses only the open cylinder. Selection uses same topology/transforms with radial1.08 overlay. Frame replacement disposes InstancedMesh buffers, deduplicates material/geometry disposal and clears superseded highlight geometry. Committed geometry tests and browser harness were inspected and executed; independent visual scrutiny and measured results are recorded below.

## Acceptance mapping

Every listed result below combines source review with the checks actually executed in this round; tests ran against pinned integration5514d99a565b1911c2a8a204e885ebd37ad6476a. The five issue diffs were reviewed at their pinned SHAs above. Browser harnesses came from their respective issue worktrees. No integration-only change is attributed as authored by a dependent issue.

### Issue20 / PR27

| Original criterion | Verification |
| --- | --- |
| Expanding bounds retain world intersections | Reference-grid unit sequence and executed browser matrix/projection assertions; initial/grown screenshots visually inspected. |
| Translation does not translate grid | Unit XYZ translation and browser moved-cell fixture compare exact grid transform. |
| Z motion does not move plane | Unit and browser negative/positive-Z frames; independent cross-feature probe rotates/translates Z while grid remains exact. |
| Division/removal/temporarily empty preserve grid | Unit and browser add/remove/restore cells; initial-empty fallback covered. |
| Orbit/pan/zoom/Fit preserve spacing | Executed browser real input gestures and Fit; grid world transform unchanged. |
| Another dataset initializes its grid | Executed browser file input and finite device fixture; actual recording reopen in independent replay probe resets dataset state. |
| Infinite planes do not cause infinite extent | Unit and executed browser plane at1e30 with empty frame retains finite ten-unit grid. |
| Initialization and transitions tested | Focused unit suite includes both reference-grid and presentation-state files; browser test passed. |
| Growing colony/stationary camera removes scale illusion | Exact grid/camera/projected-origin assertions and independent visual inspection of initial/grown screenshots support the claim. |

### Issue15 / PR30

| Original criterion | Verification |
| --- | --- |
| Fixed concentration has same RGB across frames/slices | Executed scalar browser checks actual cell instance RGB and signal texture RGBA. |
| Shared2 across [2,4]/[2,40] | Unit and executed browser both use these frames under [0,10]. |
| Endpoint colors preserve source/inspector | Executed browser clips -5/50 while renderer source and inspector retain -5/50. |
| Empty and constant do not yield NaN/misleading legend | Units and browser verify midpoint constant color, explicit constant/no-values labels and empty automatic dashes. |
| Invalid input preserves last valid config/actionable message | Browser checks reversed, Infinity, blank; independent probe checks NaN draft/error persists through moving/reset/reverse frames with identical active colors. |
| Each channel restores own configuration | Unit key separation and browser switching species/signal indices. |
| Reset and replay settings retention | Committed live reset/reversed frames; independent real recording timeline forward/back/playback tests fixed species and signal bounds through absent/returning signals. |
| Species and signal keyboard use | Executed browser native-select type-ahead and Enter form submission. |
| Unit and browser both rendering paths | Focused unit pass plus actual WebGL buffers/texture checks. Independent ±Number.MAX_VALUE range produced [0,.5,1] and finite cell colors/signal texture. |

### Issue23 / PR29

| Original criterion | Verification |
| --- | --- |
| Four constraint types and outlines disappear together | Executed fixture has all four meshes and box/cylinder outlines; toggles common device group; source confirms sibling ownership. |
| Cells/highlight/signal/grid remain | Browser asserts sibling visibility, selected-cell highlight and exact grid transform. |
| Incoming frames retain hidden state | Executed time sequence1,20,3,0; absent/returning geometry; independent live and real replay probes. |
| Camera/simulation/constraints/checkpoints/masks untouched | Setter only assigns Group.visible; source review finds no physics/data mutation. Browser compares camera/bounds/grid before/after. No native runtime claim. |
| Fit policy unchanged, no implicit refit | Executed hidden/shown Fit comparisons have same bounds/pose; independent moving frames preserve camera. |
| Picking/inspection continue | Committed browser actual projected mouse click with visible/hidden device selects/clears/reselects. |
| Labeled/keyboard control | Executed accessible-label lookup and Space toggles. |
| State retention/browser verification | Unit presentation-state retention; browser and independent recording tests pass. Missing geometry disables checkbox without losing false preference; reopening recording resets true. |

### Issue19 / PR32

| Original criterion | Verification |
| --- | --- |
| Red-only/green-only/coexpression/zero distinct | Unit and executed browser actual instance buffers verify red,green,yellow,black fixture. |
| Full red+green yields yellow | Exact endpoint buffer/unit assertions. |
| Order independent | Canonical numerical summation; exact reversed configurations and browser reorder comparisons. |
| Disable removes only that channel | Unit and browser green disabled leaves red-only contribution. |
| Range retention through toggle/reorder | Units and executed browser; independent actual replay retained per-index bounds and order. |
| Negative/out-of-range clip without data mutation | Unit -10/40 and browser -1/2; actual picked inspector keeps original concentrations. |
| No enabled channels neutral + explicit legend | Executed buffer decoding verifies neutral gray and legend text. |
| Inspector/picking/selection/lineage intact | Executed actual picking checks cell3/parent1/species, moving selection remains; removed selected ID clears. Capsule browser separately verifies reordered stable-ID selection. |
| Live/reset/replay settings survive | Committed moving/missing/reset frames; independent timeline forward/back and playback retain composite tints,bounds,order,mode and device preference. |
| Normalization/blending/color-space/state unit coverage | Focused composite/color/scalar suite passed; unit half-linear-red maps to~.73536sRGB and mid-gray tint is decoded before multiplication. |
| Moving overlapping-expression browser | Committed browser passed with moving red/green/coexpression fixture and selected-cell assertions; screenshots generated. |

Shared-editor synchronization was additionally checked independently: invalid composite draft survives ordinary frames/missing channels, then a valid single-species change updates composite fields and clears stale invalid status. No duplicate scalar state path found.

### Issue21 / PR36

| Original criterion | Verification |
| --- | --- |
| Record baseline isolated/moving dense artifact | Independently inspected committed isolated/dense before images and original baseline moving36 PNG: cap rings visible. Baseline executable was not rerun. |
| Continuous join positions/normals/no disks/cracks | Executed capsule geometry tests compare exact local equator samples, normals, outward winding and open cylinder; before/after images show join correction. |
| Multiple lengths/radii incl zero | Executed cases(0,.5),(.0001,.08),(.2,2),(3,.5),(100,.01), including Float32 instance representation and expected bounds. |
| Arbitrary orientations | Executed X,-Y,Z,(2,-3,4) direction tests and browser prescribed yaw/tilt motion. |
| Near/distant correction same camera/light | Independently inspected original near before/after isolated+dense and newly generated far/moving views; cap-ring artifact absent. Existing baseline far evidence retained; no claim of hardware coverage. |
| Distinguish residual aliasing/motion | Documentation explicitly distinguishes remaining silhouette/pixel aliasing and prescribed motion; observed faceting is not interpreted as cap seam. |
| Color/ray picking/stable IDs | Real Three.js tip raycasts plus executed browser pointer selection and reordered ID retention/removal. |
| Highlight aligned | Executed world-surface overlay maximum error1.2836774865299105e-8; zero-length finite transforms and screenshot inspected. |
| Remains instanced | Executed browser512-cell fixture has exactly3 draws and3 shared cell geometries. |
| Before/after draws/geometry/timing recorded | Committed report/baseline evidence record before:3draws,388vertices,504triangles/cell,median2.2ms,p952.4ms. This round independently remeasured after:3draws,400vertices,576triangles/cell,median2.100000024ms,p952.399999976ms. Both software-WebGL environments; timing difference is not a performance improvement claim. |
| Replaced GPU resources released | Executed instrumented actual WebGL create/delete accounting: buffers18→18 across59 replacements,0 after empty and0 after disposal. |

## Checks actually executed

- `pnpm --dir viewer test tests/reference-grid.test.ts tests/presentation-state.test.ts tests/scalar-range.test.ts tests/color.test.ts tests/composite.test.ts tests/capsule.test.ts`: **44 tests passed / 6 files** at integration5514d99.
- `run-browser.py`: all five issue-owned browser scripts against coordinator-owned Vite4343; **all passed**, no page errors. Logs: `issue-20-browser.log`, `issue-15-browser.log`, `issue-23-browser.log`, `issue-19-browser.log`, `issue-21-browser.log`. Screenshots/videos/results under matching `issue-N-browser/` directories.
- `run-cross-feature.py` / `cross-feature.mjs`: **passed**. Independent live-frame probe for invalid-draft retention, hidden device, camera/grid stability, moving arbitrary unit orientation, missing/returning channels/device, active-config cross-editor synchronization and extreme finite ranges. Log:`cross-feature.log`; screenshot:`cross-feature-final.png`.
- `run-replay-features.py` / `replay-features.mjs`: **passed**. Independent actual recording-folder open using original exported v3 grid fixture; timeline forward/reverse, playback to end, composite tints/order/per-channel ranges and signal fixed bounds retained through missing device/signal; reopening resets defaults. Log:`replay-features.log`.
- All five issue worktrees verified clean after review. Integration server remains coordinator-owned; no reviewer-owned server requires cleanup.

## Limits and discarded candidates

All runtime tests used Chromium153.0.8010.12 on macOS, ANGLE/Vulkan SwiftShader, against integration5514d99. No Safari, Firefox or physical-GPU visual/performance coverage. Original baseline executable/benchmark was not rerun: original images/measurement artifacts were inspected and the corrected fixture was rerun independently. This review does not claim CPU/Metal/CUDA simulation validation.

The first server attempt hit sandbox EPERM; a subsequent approval wait was interrupted. Coordinator launched4343 and all browser work then completed. Independent probe setup initially used an unsupported CommonJS import for canonicalize and a nonunit direction rejected by the scene parser; probe-only corrections used the package's ESM import and a unit direction. Final replay assertion initially queried an editor inside collapsed details; opening Display settings made the final check pass. None were product defects.

The known pointerdown restoreWorldUp selection jump after ViewCube snap predates this campaign and is excluded as instructed. No new material correctness, acceptance, lifecycle or presentation-state defect was confirmed in the five reviewed PRs.
