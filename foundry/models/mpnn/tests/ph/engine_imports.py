"""Import the PH engine module without its heavy third-party stack.

The engine module imports atomworks, biotite and friends at module level, although the
code under test (scorer, objective helpers, block descent) is pure torch. This helper
stubs a package only when *project code* (``mpnn`` or ``foundry``) asks for one that is
not installed. Installed packages are always real, and imports made by third-party code
such as torch get the genuine ``ImportError``, so the same tests run against the real
stack once the environment exists and nothing is stubbed then.

Stubbed CamelCase names are plain classes (so project classes can subclass them); other
names are ``MagicMock`` objects. Nothing the stubs return may be relied on by a test.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ENGINE_MODULE = "mpnn.inference_engines.potts_mpnn_ph"
PROJECT_ROOTS = {"mpnn", "foundry"}
FOUNDRY_SRC = Path(__file__).resolve().parents[4] / "src"


def _stub_class(name: str) -> type:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return MagicMock(name=f"{name}.{attr}")

    return type(name, (), {"__init__": __init__, "__getattr__": __getattr__})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name[:1].isupper() and not name.isupper():
            value = _stub_class(name)
        else:
            value = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


def _requested_by_project() -> bool:
    """True when the import in progress was written in ``mpnn`` or ``foundry`` code."""
    frame = sys._getframe(2)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if not module.startswith("importlib"):
            return module.split(".")[0] in PROJECT_ROOTS
        frame = frame.f_back
    return False


class _ProjectStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Last in line: reached only when no real finder located the module."""

    def __init__(self):
        self.created: list[str] = []

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in PROJECT_ROOTS or not _requested_by_project():
            return None
        self.created.append(name)
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


@contextlib.contextmanager
def engine_module():
    """Yield the imported engine module; undo all stubbing and imports on exit."""
    before = set(sys.modules)
    path_before = list(sys.path)
    if importlib.util.find_spec("foundry") is None and FOUNDRY_SRC.exists():
        sys.path.append(str(FOUNDRY_SRC))
    finder = _ProjectStubFinder()
    sys.meta_path.append(finder)
    try:
        yield importlib.import_module(ENGINE_MODULE)
    finally:
        sys.meta_path.remove(finder)
        sys.path[:] = path_before
        # Drop only what the project and the stubs brought in: removing real third-party
        # modules (torch internals) would make a re-import fail.
        owned = PROJECT_ROOTS | {n.split(".")[0] for n in finder.created}
        for name in set(sys.modules) - before:
            if name.split(".")[0] in owned:
                del sys.modules[name]
