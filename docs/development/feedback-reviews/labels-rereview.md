# Independent re-review: issue 18 / PR 28 channel label fix

This is an archived independent review. See the [round summary](../feedback-review.md) for final heads, corrections and combined validation. Temporary evidence paths in this report identify local artifacts; they are not repository dependencies.

Verdict: **approve-as-fixed**. No substantive remaining finding in the bounded correction.

Reviewed `ae428e5dae170e3e58b54804c3b39b90c07f63bf..98382896f91ab9ffb349567683041097e7f477e6` (`f450e13`, `9838289`) against `/private/tmp/microsimulator-swarm/assignments/issue-18.md`. HEAD and `marpaia/18` both resolve to the reviewed final commit. The diff contains only the four requested files, and the worktree remains clean. This is an independent agent review, not a formal GitHub approval.

## Correctness and ownership

- `viewer/src/scene.ts:116`: ASCII normalization occurs before duplicate counting. Tabs, line feeds, form feeds, carriage returns and ordinary spaces collapse to one space with edge trimming. Blank labels retain index-based fallbacks, while nonblank Unicode and HTML-like text remain text. Normalization changes only newly created presentation strings, not decoded metadata or serialized source.
- Ordinary duplicates retain the usual index suffix. If a generated label collides with a literal label, indexing every member of the group is sufficient: each final string ends with its distinct numerical index in brackets. Two different indices cannot have identical such endings, even when the original names contain arbitrary nested suffixes. Fallback-name collisions use the same rule.
- `viewer/src/scene.ts:110`: the WeakMap key is the readonly group array. The computed result depends only on array contents and positions, so reuse across frames or species/signals is valid when the same immutable array is shared. Cache values contain only strings and do not retain their weak keys or scene frames.
- Decoder ownership supports the cache assumption: `parseChannelMetadata` maps input entries into fresh arrays, legacy decoding creates fresh fallback arrays, and live frames go through `parseScene`. Source inspection found no production mutation of `channelMetadata`. Replacement frames therefore get independently resolved labels. Runtime arrays are not frozen, but in-place mutation would violate the existing readonly contract; there is no supported production mutation path requiring invalidation.
- Numerical channel identity is unchanged. The correction does not touch selector values, channel counts, numerical cell data, color configuration or persistence. Existing callers use this formatter for selectors, legends and the inspector; selector options continue to use numerical values.

## Regression coverage

The committed additions meaningfully exercise both species and signals, including raw duplicate names, fallback collisions, nested generated suffixes, and combined whitespace/suffix collisions. The browser addition checks actual `HTMLOptionElement.label`, avoiding a whitespace-normalizing text matcher as the sole oracle, and checks selected numerical values and legend text across successive renamed frames. These assertions target the reported failures rather than only mirroring the implementation.

Independent focused execution imported the actual final `viewer/src/scene.ts` with Node 24.1.0 and passed **4,412 group checks**, including exhaustive testing of **2,197 triples** drawn from ordinary names, blank/Unicode labels, whitespace variants, fallbacks and generated suffixes. Additional explicit checks cover all five ASCII whitespace characters, nested Unicode suffix collisions, preservation of nonblank Unicode/HTML-like strings and raw metadata, repeated cached reads, shared frozen group arrays, fresh decoded array ownership, independent renamed frames, unchanged cell data and range validation after caching.

Reproduction: `node /private/tmp/microsimulator-swarm/review-round-1/coordinator/labels-rereview-probe.mjs`.

Artifacts: `labels-rereview-probe.mjs` and `labels-rereview-probe.log` in this directory. The final probe passed; an initial probe-only CommonJS import of the ESM-only canonicalize package failed and was corrected before the successful run. No production or committed test files were edited.

The coordinator's `labels-fix.md` records four regressions failing before the corrections and 43 viewer tests, TypeScript/build, and real Chromium passing afterward. Those full-suite/browser runs were not repeated in this bounded review. I inspected the committed browser assertions and performed the focused independent checks above. Garbage collection behavior was assessed from WeakMap ownership rather than a nondeterministic collection test. No new browser server, GitHub post, merge or branch change was made.
