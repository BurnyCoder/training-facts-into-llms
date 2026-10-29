"""Build derived README/website views through separate offline evidence phases.

The scientific reports are inputs only. The explicit output inventory prevents
private logs, checkpoints, or configuration from entering the Pages bundle.
Source: https://docs.python.org/3/library/pathlib.html
"""

from __future__ import annotations

import json
import re
import shutil
import stat
from pathlib import Path

from training_facts_into_llms.results_evidence import load_results
from training_facts_into_llms.results_render import chart_files, readme_blocks

SITE_FILES = ("index.html", "app.js", "styles.css")


def _safe_path(root: Path, relative: str) -> Path:
    """Reject output aliases and non-regular files before any tracked write."""
    path = root / relative
    if not path.is_relative_to(root) or ".." in Path(relative).parts:
        raise ValueError("result path escapes repository")
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError(f"result path contains a symlink: {relative}")
        if candidate != path and candidate.exists() and not candidate.is_dir():
            raise ValueError(f"result parent is not a directory: {relative}")
    if path.exists():
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"result file is not a single regular file: {relative}")
    return path


def _replace_blocks(readme: str, blocks: dict[str, str]) -> str:
    """Require exactly one ordered marker pair per generated section."""
    for name, content in blocks.items():
        start, end = (f"<!-- results:{name}:{edge} -->" for edge in ("start", "end"))
        if readme.count(start) != 1 or readme.count(end) != 1:
            raise ValueError(f"missing or repeated results marker: {name}")
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
        if not pattern.search(readme):
            raise ValueError(f"reversed results markers: {name}")
        replacement = f"{start}\n{content}\n{end}"
        readme = pattern.sub(lambda _, value=replacement: value, readme)
    return readme


def build_results(root: Path, *, check: bool = False) -> list[str]:
    """Validate inputs, render all views, then check freshness or write the bundle."""
    root = root.resolve()
    data = load_results(root)
    # Finish evidence and marker validation before touching any generated output.
    readme = _safe_path(root, "README.md").read_text(encoding="utf-8")
    rendered_readme = _replace_blocks(readme, readme_blocks(data))
    outputs = {
        "README.md": rendered_readme.encode(),
        "site/results.json": (
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode(),
        **chart_files(root, data),
    }
    paths = {name: _safe_path(root, name) for name in outputs}
    static = {
        name: _safe_path(root, f"site/{name}").read_bytes() for name in SITE_FILES
    }
    # A bundle is assembled from this inventory, never a recursive repository copy.
    bundle = {**static, "results.json": outputs["site/results.json"], ".nojekyll": b""}
    bundle.update(
        {
            "assets/" + Path(name).name: value
            for name, value in outputs.items()
            if name.startswith("docs/assets/")
        }
    )
    if not check:
        for name in bundle:
            _safe_path(root, "artifacts/results-site/" + name)
        destination = root / "artifacts/results-site"
        if destination.exists():
            for path in destination.rglob("*"):
                if path.is_symlink() or (
                    path.is_file()
                    and path.relative_to(destination).as_posix() not in bundle
                ):
                    raise ValueError(
                        "unexpected file in reserved results-site directory"
                    )
    changed = [
        name
        for name, value in outputs.items()
        if not paths[name].exists() or paths[name].read_bytes() != value
    ]
    if check:
        if changed:
            raise ValueError("stale generated results: " + ", ".join(changed))
        return []
    for name in changed:
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_bytes(outputs[name])
    destination = root / "artifacts/results-site"
    # The directory is reserved for this derived bundle and contains no evidence.
    if destination.exists():
        shutil.rmtree(destination)
    for name, value in bundle.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return changed
