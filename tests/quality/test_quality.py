"""Behavioral coverage for source discovery, paragraph spacing, and strict limits."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import quality_files
from check_complexity import violations
from code_paragraphs import format_code


class ParagraphTests(unittest.TestCase):
    def test_setup_guard_computation_and_return(self) -> None:
        code = """int count(int n) {
  int result = 0;
  for (int i = 0; i < n; ++i) {
    const int x = i * i;
    if (x == 0) {
      continue;
    }
    result += x;
  }
  return result;
}
"""
        result = format_code(Path("kernel.cu"), code)
        self.assertIn("result = 0;\n\n  for", result)
        self.assertIn("i * i;\n\n    if", result)
        self.assertIn("continue;\n    }\n\n    result", result)
        self.assertIn("}\n\n  return", result)
        self.assertEqual(format_code(Path("kernel.cu"), result), result)
        self.assertEqual("".join(code.split()), "".join(result.split()))

    def test_comments_chains_and_string_literals(self) -> None:
        code = """int f(int x) {
  const char* s = R"tag(for (;;) {
    if (x) { return x; }
  })tag";
  // Keep this comment attached to the guard.
  if (x) {
    return 1;
  } else {
    return 2;
  }
}
"""
        result = format_code(Path("source.cpp"), code)
        self.assertIn('})tag";\n\n  // Keep', result)
        self.assertIn("guard.\n  if", result)
        self.assertIn("} else {", result)
        self.assertIn("if (x) { return x; }", result)

    def test_preprocessor_and_empty_bodies_are_opaque(self) -> None:
        code = "#define LOOP(x) do { x++; if (x) break; } while (0)\nvoid f() {}\n"
        self.assertEqual(format_code(Path("header.hpp"), code), code)

    def test_python_and_typescript(self) -> None:
        cases = {
            "source.py": "def f(x):\n    y = x + 1\n    if y:\n        y += 1\n    return y\n",
            "source.ts": (
                "function f(x: number) {\n  let y = x + 1;\n"
                "  if (y) {\n    y++;\n  }\n  return y;\n}\n"
            ),
        }

        for name, code in cases.items():
            with self.subTest(name=name):
                result = format_code(Path(name), code)
                self.assertGreater(result.count("\n\n"), 1)
                self.assertEqual(format_code(Path(name), result), result)


class ComplexityTests(unittest.TestCase):
    def test_line_boundary(self) -> None:
        for count, fails in [(98, False), (99, True)]:
            # Python definition and return each contribute one code line.
            code = "def f():\n" + "    x = 1\n" * count + "    return x\n"
            self.assertEqual(bool(violations(Path("source.py"), code)), fails)

    def test_complexity_boundary_and_gpu_extensions(self) -> None:
        for extension in ["cpp", "hpp", "cu", "cuh", "metal", "mm"]:
            for branches, fails in [(14, False), (15, True)]:
                with self.subTest(extension=extension, branches=branches):
                    code = "int f(int x) {\n" + "if (x) { x--; }\n" * branches + "return x;\n}\n"
                    self.assertEqual(bool(violations(Path(f"source.{extension}"), code)), fails)

    def test_templates_lambdas_and_suppressions(self) -> None:
        code = "template <class T> T f(T x) { return [](T y) { return y; }(x); }"
        self.assertEqual(violations(Path("source.hpp"), code), [])
        self.assertTrue(violations(Path("source.py"), "# lizard forgives\ndef f(): pass\n"))


class DiscoveryTests(unittest.TestCase):
    def test_tracked_untracked_and_ignored_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("ignored.py\n")

            for name in ["tracked.cpp", "new file.metal", "ignored.py", "uv.lock"]:
                (root / name).write_text("\n")

            subprocess.run(["git", "add", "tracked.cpp"], cwd=root, check=True)
            (root / "src").mkdir()
            (root / "src/child.cpp").write_text("\n")
            (root / "src/ignored.py").write_text("\n")

            with patch.object(quality_files, "ROOT", root):
                files = quality_files.source_files()
                self.assertIn(Path("tracked.cpp"), files)
                self.assertIn(Path("new file.metal"), files)
                self.assertNotIn(Path("ignored.py"), files)
                self.assertNotIn(Path("uv.lock"), files)
                self.assertEqual(
                    quality_files.source_files(["new file.metal"]), [Path("new file.metal")]
                )
                self.assertEqual(quality_files.source_files(["src"]), [Path("src/child.cpp")])
                self.assertEqual(quality_files.source_files(["."]), files)


class CommandTests(unittest.TestCase):
    def test_check_is_read_only_and_format_is_repeatable(self) -> None:
        root = quality_files.ROOT

        with tempfile.TemporaryDirectory(dir=root) as directory:
            source = Path(directory) / "example.cpp"
            source.write_text("int f(int x){int y=x+1;if(y){y++;}return y;}\n")
            original = source.read_bytes()
            command = [sys.executable, str(root / "scripts/quality.py")]
            selected = str(source.relative_to(root))
            check = subprocess.run(
                [*command, "check", selected], cwd=root, capture_output=True, text=True
            )
            self.assertNotEqual(check.returncode, 0)
            self.assertEqual(source.read_bytes(), original)

            previous = None

            for _ in range(2):
                result = subprocess.run(
                    [*command, "format", selected], cwd=root, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                formatted = source.read_bytes()

                if previous is not None:
                    self.assertEqual(formatted, previous)

                previous = formatted
                self.assertIn(b"\n\n  if", formatted)
                check = subprocess.run(
                    [*command, "check", selected], cwd=root, capture_output=True, text=True
                )
                self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
                self.assertEqual(source.read_bytes(), formatted)

            self.assertEqual(source.read_bytes(), formatted)


if __name__ == "__main__":
    unittest.main()
