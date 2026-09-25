#!/usr/bin/env python3
"""Stage 19 — MECHANICAL proof that the held-out guard has one call-site.

Protocol v1.1 section 5.4 promises that the test split is opened exactly once
and that the opt-in variable can only be set from
scripts/evaluation/run_evaluation.py. A promise that only lives in prose is
not a guarantee, so this test greps the source tree for two different things:

  1. MENTIONS of the literal, anywhere in scripts/ (which holds evaluation/,
     training/, analysis/, simulation/ and cluster/), dataset/, models/,
     results/ and the repository root. Only an allow-list of files may name
     it at all; anything else is a failure, because a file that knows the
     name is a file that could set it tomorrow. dataset/, models/ and
     results/ are in the sweep because that is where a helper that "just
     needs the test shards" would naturally be written, and the repository
     root because a one-off script dropped there would have been invisible
     to a directory-only scan.
  2. ASSIGNMENTS, by any spelling that actually writes the environment:
     os.environ[...] = , os.environ.setdefault and os.putenv in Python; and in
     shell scripts `export NAME`, a leading `NAME=`, `env NAME=value cmd`, and
     `--export=...,NAME[=value]`. The last two matter here specifically: an
     `sbatch --export=ALL,<the variable>=1` line opens the held-out split from
     the submission line without any file in the tree containing an assignment
     an earlier version of this test would have recognized. Exactly one
     assignment may exist, it must be in the driver, and it must sit inside
     resolve_geometries -- after the access log has been written and behind the
     split check.

This file assembles the literal at runtime from four fragments, so its own
source never contains it and it cannot whitelist itself by accident.

Run:  python scripts/evaluation/test_single_callsite.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# assembled at runtime: the literal must not appear in this file's source
ENV_NAME = "MUOGRAPHY" + "_" + "ALLOW" + "_" + "TEST"
CONSTANT_NAME = "ALLOW_TEST_ENV"

SEARCH_DIRS = ("scripts", "dataset", "models", "results")
SEARCH_SUFFIXES = (".py", ".sh", ".sbatch")
SHELL_SUFFIXES = (".sh", ".sbatch")

# Files permitted to NAME the variable, and why. Not every entry has to
# mention it (training/test_dataset.py goes through the constant instead);
# the list is a permission, not an obligation.
ALLOWED_MENTIONS = {
    "scripts/training/muography_dataset.py":
        "defines the constant and raises on the held-out splits",
    "scripts/evaluation/run_evaluation.py":
        "the single authorized call-site (Stage 20 driver)",
    "scripts/training/test_dataset.py":
        "tests the guard itself",
    "scripts/evaluation/test_single_callsite.py":
        "this test (assembles the literal at runtime)",
}
ASSIGNMENT_ALLOWED = "scripts/evaluation/run_evaluation.py"

_TARGET = rf"""(?:["']{ENV_NAME}["']|{CONSTANT_NAME})"""
PY_ASSIGNMENT = re.compile(
    rf"""os\.environ\s*\[\s*{_TARGET}\s*\]\s*=(?!=)"""
    rf"""|os\.environ\.setdefault\s*\(\s*{_TARGET}"""
    rf"""|os\.putenv\s*\(\s*{_TARGET}""")
SH_ASSIGNMENT = re.compile(
    rf"""export\s+{ENV_NAME}"""
    rf"""|(?:^|;|&&|\|\|)\s*{ENV_NAME}\s*="""
    # sbatch/srun submission-line export: --export=ALL,NAME=1 or --export=NAME
    rf"""|--export=\S*\b{ENV_NAME}\b"""
    # env NAME=value command, including env -u FOO NAME=value command
    rf"""|\benv\s+[^\n]*?\b{ENV_NAME}\s*=""")


def _sources():
    for directory in SEARCH_DIRS:
        root = REPO_ROOT / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix in SEARCH_SUFFIXES and "__pycache__" not in path.parts:
                yield path
    # a script dropped in the repository root is as able to open the split as
    # one filed under a directory the sweep happens to list
    for path in sorted(REPO_ROOT.glob("*.py")):
        if path.is_file():
            yield path


def _key(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _read(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def test_mentions() -> bool:
    """No file outside the allow-list may contain the literal at all."""
    offenders, seen = [], {}
    for path in _sources():
        text = _read(path)
        if text is None or ENV_NAME not in text:
            continue
        key = _key(path)
        seen[key] = text.count(ENV_NAME)
        if key not in ALLOWED_MENTIONS:
            offenders.append(key)
    ok = not offenders
    for key in sorted(seen):
        print(f"    {key}: {seen[key]} mention(s) — "
              f"{ALLOWED_MENTIONS.get(key, 'NOT ALLOWED')}")
    if offenders:
        print(f"    unexpected files naming the guard: {offenders} -> FAIL")
    print(f"  mentions: {len(seen)} file(s) scanned as allowed, "
          f"{len(offenders)} unexpected -> {'ok' if ok else 'FAIL'}")
    return ok


def find_assignments():
    """[(path key, line number, line)] over every spelling that writes the env."""
    found = []
    for path in _sources():
        text = _read(path)
        if text is None:
            continue
        key = _key(path)
        if key == "scripts/evaluation/test_single_callsite.py":
            continue                    # the regex source is not an assignment
        pattern = (SH_ASSIGNMENT if path.suffix in SHELL_SUFFIXES
                   else PY_ASSIGNMENT)
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                found.append((key, number, line.strip()))
    return found


def test_assignments() -> bool:
    """Exactly one assignment in the whole tree, and it is the driver's."""
    found = find_assignments()
    outside = [entry for entry in found if entry[0] != ASSIGNMENT_ALLOWED]
    inside = [entry for entry in found if entry[0] == ASSIGNMENT_ALLOWED]
    checks = {
        "no assignment outside the driver": not outside,
        "exactly one assignment inside the driver": len(inside) == 1,
    }
    for key, number, line in outside:
        print(f"    assignment outside the driver: {key}:{number}: {line}")
    for key, number, line in inside:
        print(f"    authorized: {key}:{number}: {line}")
    ok = all(checks.values())
    print(f"  assignments: {len(found)} total, {len(outside)} unauthorized -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_driver_shape() -> bool:
    """The one assignment must sit in resolve_geometries, behind the guards."""
    path = REPO_ROOT / ASSIGNMENT_ALLOWED
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    numbers = [n for n, line in enumerate(lines, start=1)
               if PY_ASSIGNMENT.search(line)]
    checks = {"exactly one assignment line": len(numbers) == 1}
    if numbers:
        target = numbers[0]
        enclosing = ""
        for n in range(target - 1, 0, -1):
            if lines[n - 1].startswith("def "):
                enclosing = lines[n - 1]
                break
        preceding = "\n".join(lines[:target])
        checks.update({
            "the enclosing function is resolve_geometries":
                enclosing.startswith("def resolve_geometries"),
            "the pre-set guard is checked first":
                "assert_guard_not_preset()" in preceding,
            "the access log is written first":
                "access_log_open(" in preceding,
            "the assignment is behind the split check":
                'split == "test"' in preceding,
        })
    checks["the driver exports a refusal helper"] = (
        "def assert_guard_not_preset" in text)
    checks["the driver snapshots the environment at import"] = (
        "GUARD_PRESET_AT_IMPORT" in text)
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  driver shape: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19 single-call-site test")
    results = {
        "mentions": test_mentions(),
        "assignments": test_assignments(),
        "driver_shape": test_driver_shape(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 SINGLE CALL-SITE:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
