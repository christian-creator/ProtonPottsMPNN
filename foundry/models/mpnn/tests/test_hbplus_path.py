"""HBPLUS_PATH is checked before HBPLUS is run, and the errors say what to do."""

import os
from pathlib import Path

import pytest

pytest.importorskip("atomworks")


@pytest.fixture(scope="module")
def hbplus_executable():
    from mpnn.transforms.bond_annotation import hbplus_executable

    return hbplus_executable


def make_executable(path: Path, mode=0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(mode)
    return path


@pytest.mark.parametrize("value", [None, ""])
def test_an_unset_path_points_to_the_readme(hbplus_executable, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("HBPLUS_PATH", raising=False)
    else:
        monkeypatch.setenv("HBPLUS_PATH", value)
    with pytest.raises(ValueError, match="Getting HBPLUS"):
        hbplus_executable()


def test_a_missing_file_or_a_directory_is_not_accepted(
    hbplus_executable, monkeypatch, tmp_path
):
    monkeypatch.setenv("HBPLUS_PATH", str(tmp_path / "nowhere" / "hbplus"))
    with pytest.raises(FileNotFoundError, match="not a file"):
        hbplus_executable()
    monkeypatch.setenv("HBPLUS_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not the directory"):
        hbplus_executable()


def test_a_file_without_execute_permission_is_not_accepted(
    hbplus_executable, monkeypatch, tmp_path
):
    monkeypatch.setenv("HBPLUS_PATH", str(make_executable(tmp_path / "hbplus", 0o644)))
    with pytest.raises(PermissionError, match="not executable"):
        hbplus_executable()


def test_a_relative_path_is_made_absolute(hbplus_executable, monkeypatch, tmp_path):
    """HBPLUS runs in a temporary directory, where a relative path would no longer resolve."""
    target = make_executable(tmp_path / "bin" / "hbplus")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HBPLUS_PATH", os.path.join("bin", "hbplus"))
    found = Path(hbplus_executable())
    assert found.is_absolute() and found.samefile(target)
