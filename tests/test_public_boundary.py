"""The public framework must never depend on the private overlay.

``swingml/proprietary/`` is excluded from the published tree. If a module that
*is* published imported from it, the published release would be unrunnable -- and
the failure would only surface for someone who cloned it, not here. So the
boundary is enforced by a test rather than by convention.

Dependencies run one way only: ``proprietary`` may import public modules, but
nothing public may import ``proprietary``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "swingml"
PRIVATE_DIR = PACKAGE_ROOT / "proprietary"
PRIVATE_PACKAGE = "swingml.proprietary"


def _public_source_files() -> list[Path]:
    """Every module that ships publicly (i.e. outside the private overlay)."""
    return sorted(
        p for p in PACKAGE_ROOT.rglob("*.py")
        if PRIVATE_DIR not in p.parents
    )


def test_public_source_files_found():
    """Guard the guard: a bad glob would make every other check vacuous."""
    files = _public_source_files()
    assert len(files) >= 8, f"expected the public package to have modules, found {files}"
    assert all(PRIVATE_DIR not in p.parents for p in files)
    # The packages that must always ship, whatever the deployment.
    names = {p.name for p in files}
    for required in ("base.py", "registry.py", "demo.py", "evaluation.py", "validation.py"):
        assert required in names, f"public tree is missing {required}"


def test_no_public_module_imports_the_private_overlay():
    offenders: list[str] = []

    for path in _public_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == PRIVATE_PACKAGE or name.startswith(PRIVATE_PACKAGE + "."):
                    rel = path.relative_to(PACKAGE_ROOT.parent)
                    offenders.append(f"{rel}:{node.lineno} imports {name}")

    assert not offenders, (
        "public modules must not import the private overlay -- they would be "
        "broken in the published tree:\n  " + "\n  ".join(offenders)
    )


def test_private_overlay_is_a_recognisable_package():
    """Where the overlay exists, the publish script must be able to key on it.

    In the published tree the overlay is absent by construction, so this is a
    private-repo check that skips rather than fails there.
    """
    if not PRIVATE_DIR.exists():
        pytest.skip("private overlay not present (published tree) -- nothing to check")
    assert (PRIVATE_DIR / "__init__.py").exists(), (
        "the overlay must stay an importable package, or the publish script's "
        "path matching would silently stop removing it"
    )
    assert _public_source_files(), "private dir must not swallow the whole package"


def test_importing_public_modules_does_not_pull_in_private_code():
    """Run in a subprocess: this test suite itself imports the overlay elsewhere,
    so only a fresh interpreter can prove the public modules stand alone."""
    code = (
        "import sys;"
        "import swingml.features, swingml.dataset, swingml.cli;"
        "leaked = sorted(m for m in sys.modules"
        " if m == 'swingml.proprietary' or m.startswith('swingml.proprietary.'));"
        "print('|'.join(leaked))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(PACKAGE_ROOT.parent),
    )
    assert proc.returncode == 0, f"importing public modules failed:\n{proc.stderr}"
    assert proc.stdout.strip() == "", (
        f"public imports transitively pulled in private modules: {proc.stdout.strip()}"
    )
