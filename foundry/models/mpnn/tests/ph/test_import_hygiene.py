"""The design path must not import optional packages at module level."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "mpnn"


def module_level_imports(path):
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("package", ["propka"])
def test_optional_packages_are_not_imported_at_module_level(package):
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if package in module_level_imports(path)
    ]
    assert offenders == []
