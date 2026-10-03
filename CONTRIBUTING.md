# Contributing

## Formatting and checks

Install the viewer's pinned tools with `pnpm --dir viewer install --frozen-lockfile`. The quality commands install their locked Python tools in an isolated environment and do not build the simulator or replace the development environment.

```console
uv run --isolated --locked --only-group quality python scripts/quality.py format
uv run --isolated --locked --only-group quality python scripts/quality.py check
uv run --isolated --locked --only-group quality python scripts/quality.py complexity
uv run --isolated --locked --only-group quality python -m unittest discover -s tests/quality
```

Append repository-relative filenames or directories to any quality command to select files. Full runs include tracked files and nonignored new files. `format` applies safe Ruff fixes and formatting; `check` never rewrites files. Install commit hooks with `uv run --isolated --locked --only-group quality pre-commit install`. Hooks use the same commands on staged files; CI checks the entire maintained tree. The `quality` status is the required merge check to enable in repository branch protection.

clang-format handles native code, including CUDA and Metal. Ruff handles Python lint and formatting. Prettier handles viewer code, JSON, YAML, HTML, CSS, and Markdown; it preserves prose wrapping and leaves embedded code to the appropriate language formatter. Taplo, shfmt, and cmake-format handle TOML, shell, and CMake. Generated output, dependency directories, and package-manager lockfiles are excluded.

Python tools are pinned in the `quality` dependency group and `uv.lock`; Prettier is pinned in the viewer manifest and lockfile. Upgrade tools explicitly, regenerate locks, run formatting, and review the resulting diff. Use the same pinned versions in editor format-on-save settings, with repository configuration enabled. `.editorconfig` supplies whitespace defaults.

## Readable code

Write one coherent operation per paragraph of code. Separate validation, setup, computation, and result construction with a blank line. Keep statements that jointly perform one operation together. Explain numerical assumptions and invariants in comments; avoid comments that merely repeat the code.

This applies to short functions too. Give a loop breathing room after setup, separate a guard from the operation it protects, and separate the final return from the completed computation. For example:

```cpp
std::vector<BodyVector> intersect_polygon(std::vector<BodyVector> p, const ConvexPolygon& q) {
  const auto n = normal(q);
  const double epsilon = polygon_epsilon(q.vertices);

  for (std::size_t i = 0; i < q.vertices.size() && !p.empty(); ++i) {
    const auto& a = q.vertices[i];
    const auto& b = q.vertices[(i + 1) % q.vertices.size()];
    const auto edge = body_difference(b, a);

    if (body_dot(edge, edge) <= epsilon * epsilon) {
      continue;
    }

    const auto outward = body_normalized(body_cross(edge, n));
    p = clip_polygon(p, {outward, body_dot(outward, a), 0});
  }

  return p;
}
```

The shared formatter applies this spacing mechanically using syntax trees: it inserts a blank line before loops and conditionals, after completed control-flow statements, and before a return that follows other work. It keeps attached comments with their statements and leaves one-line bodies, strings, macros, and unsupported syntax alone. `else` and `catch` remain attached to their statements. The check command enforces the same spacing in hooks and CI. Readability review still checks semantic phases within uninterrupted calculations, which syntax alone cannot identify.

Native code uses two-space indentation, 100 columns, explicit control-flow braces, and multiline nonempty function and lambda bodies. Python uses four-space indentation and 100 columns. Review brace insertion as a code change, particularly around macros and preprocessor branches.

Every native, Python, JavaScript, and TypeScript function must have at most 100 nonblank, noncomment lines and cyclomatic complexity at most 15, as measured by pinned Lizard. The same limits apply to tests, examples, bindings, and GPU kernels. Suppressions and inherited exception lists are not permitted. The structural check maps CUDA and Metal to the C++ reader and reports the original file and line.

Extract helpers around meaningful operations. Keep their inputs and outputs explicit, keep ownership local, and preserve arithmetic order, device execution, exception behavior, and state transitions. Do not hide a large function in nested functions or manufacture parameter objects just to satisfy a metric. Split long tests by scenario and binding registration by public feature. A passing metric is a review aid, not proof of clarity.

## Behavioral validation

Run `uv run --locked pyright`, the Python tests, and `pnpm --dir viewer check`, `pnpm --dir viewer test`, and `pnpm --dir viewer build` for affected application code. Native changes also require the relevant build and conformance tests described in [testing and validation](docs/development/validation.md). CUDA compilation does not establish CUDA runtime behavior.

The existing `.clang-tidy` configuration is available for deeper analysis against a build's compile database; its broad diagnostic set is separate from the shared formatting and complexity gate.
