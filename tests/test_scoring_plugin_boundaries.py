"""Exercise real Git/source and exception boundaries for trusted scoring plugins."""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from training_facts_into_llms.scoring_loader import load_scoring_plugin


def _module_name(root: Path, label: str) -> str:
    """Return a deterministic import-safe name unique to one temporary repository."""
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    return f"test_plugin_{label}_{digest}"


def _initialize_repository(root: Path) -> None:
    """Create the minimum Git index required by the production source validator."""
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


def _write_module(root: Path, module_name: str, source: str, *, tracked: bool) -> Path:
    """Write one temporary plugin and optionally place its exact bytes in the index."""
    module_path = root / f"{module_name}.py"
    module_path.write_text(source, encoding="utf-8")
    if tracked:
        subprocess.run(["git", "add", module_path.name], cwd=root, check=True)
    importlib.invalidate_caches()
    return module_path


def test_plugin_loader_rejects_real_source_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An importable external module cannot cross the repository trust boundary."""
    root = tmp_path / "repository"
    external = tmp_path / "external"
    _initialize_repository(root)
    external.mkdir()
    module_name = _module_name(root, "outside")
    _write_module(external, module_name, "def create(*args):\n    return None\n", tracked=False)
    monkeypatch.syspath_prepend(str(external))

    with pytest.raises(ValueError, match="inside the repository"):
        load_scoring_plugin(root, f"{module_name}:create")


def test_plugin_loader_rejects_real_untracked_repository_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repository containment alone cannot authorize unreviewed executable code."""
    root = tmp_path / "repository"
    _initialize_repository(root)
    module_name = _module_name(root, "untracked")
    _write_module(root, module_name, "def create(*args):\n    return None\n", tracked=False)
    monkeypatch.syspath_prepend(str(root))

    with pytest.raises(ValueError, match="tracked by Git"):
        load_scoring_plugin(root, f"{module_name}:create")


def test_dotted_plugin_parent_is_audited_without_executing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source discovery cannot execute an untracked dotted package initializer."""
    root = tmp_path / "repository"
    _initialize_repository(root)
    package_name = _module_name(root, "package")
    package = root / package_name
    package.mkdir()
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "Path('parent-executed.sentinel').write_text('executed')\n",
        encoding="utf-8",
    )
    (package / "plugin.py").write_text(
        "def create(*args):\n    return None\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", f"{package_name}/plugin.py"], cwd=root, check=True)
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()

    with pytest.raises(ValueError, match="tracked by Git"):
        load_scoring_plugin(root, f"{package_name}.plugin:create")

    assert not (root / "parent-executed.sentinel").exists()


def test_tracked_plugin_factory_exception_aborts_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory errors remain fatal and cannot silently select a fallback scorer."""
    root = tmp_path / "repository"
    _initialize_repository(root)
    module_name = _module_name(root, "factory")
    _write_module(
        root,
        module_name,
        "def create(*args):\n    raise RuntimeError('factory failed')\n",
        tracked=True,
    )
    monkeypatch.syspath_prepend(str(root))

    with pytest.raises(RuntimeError, match="factory failed"):
        load_scoring_plugin(root, f"{module_name}:create")

    sys.modules.pop(module_name, None)


def test_tracked_plugin_score_and_decide_exceptions_abort_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted plugin execution errors propagate instead of producing partial evidence."""
    root = tmp_path / "repository"
    _initialize_repository(root)
    module_name = _module_name(root, "runtime")
    _write_module(
        root,
        module_name,
        """
class Plugin:
    def score(self, cases, generations, *, phase):
        raise RuntimeError("score failed")

    def decide(self, baseline, tuned):
        raise RuntimeError("decide failed")

def create(scoring_options, acceptance_options):
    return Plugin()
""".lstrip(),
        tracked=True,
    )
    monkeypatch.syspath_prepend(str(root))

    plugin, source = load_scoring_plugin(root, f"{module_name}:create")

    assert source == root / f"{module_name}.py"
    with pytest.raises(RuntimeError, match="score failed"):
        plugin.score([], [], phase="baseline")
    with pytest.raises(RuntimeError, match="decide failed"):
        plugin.decide(None, None)  # type: ignore[arg-type]
    sys.modules.pop(module_name, None)


def test_canonical_source_mismatch_never_executes_plugin_module_in_fresh_process(
    tmp_path: Path,
) -> None:
    """A fresh interpreter must hash canonical bytes before top-level plugin code."""
    root = tmp_path / "repository"
    package = root / "src" / "training_facts_into_llms"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    # Reuse the production dependency-free verifier in an isolated import graph.
    project_root = Path(__file__).resolve().parents[1]
    shutil.copy2(
        project_root / "src/training_facts_into_llms/scoring_loader.py",
        package / "scoring_loader.py",
    )
    (package / "scoring.py").write_text(
        """
from pathlib import Path
Path("plugin-executed.sentinel").write_text("executed", encoding="utf-8")

class Plugin:
    def score(self, cases, generations, *, phase):
        return None
    def decide(self, baseline, tuned):
        return None

def create_canonical_plugin(scoring_options, acceptance_options):
    return Plugin()
""".lstrip(),
        encoding="utf-8",
    )
    (package / "evaluation.py").write_text("# delegated scoring\n", encoding="utf-8")
    (package / "json_values.py").write_text("# JSON validation\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "src"], cwd=root, check=True)
    script = """
from pathlib import Path
from training_facts_into_llms.scoring_loader import (
    CANONICAL_PLUGIN_TARGET,
    load_scoring_plugin,
)

try:
    load_scoring_plugin(
        Path.cwd(),
        CANONICAL_PLUGIN_TARGET,
        expected_source_sha256="0" * 64,
    )
except ValueError as error:
    if "source SHA-256 differs" not in str(error):
        raise
else:
    raise AssertionError("mismatched canonical source unexpectedly loaded")
"""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (root / "plugin-executed.sentinel").exists()


def test_resolution_and_phase_binding_do_not_preimport_canonical_bundle() -> None:
    """Config and phase construction remain pure until the first verified gate."""
    project_root = Path(__file__).resolve().parents[1]
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace

from training_facts_into_llms.experiments import resolve_experiment
from training_facts_into_llms.pipeline import (
    _AttemptState,
    _GateCache,
    _build_attempt_phases,
)

root = Path.cwd()
experiment = resolve_experiment(root, "positive_primary")
state = _AttemptState(
    run_id="import-audit",
    profile=experiment.profile,
    gate_cache=_GateCache(),
    experiment=experiment,
)
_build_attempt_phases(SimpleNamespace(root=root), state)
for module in (
    "training_facts_into_llms.scoring",
    "training_facts_into_llms.evaluation",
    "training_facts_into_llms.json_values",
):
    if module in sys.modules:
        raise AssertionError(f"canonical bundle imported before verification: {module}")
"""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(project_root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
