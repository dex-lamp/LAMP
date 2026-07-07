"""AST-based Python syntax check that does not create bytecode caches."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


DEFAULT_ROOTS = ("vae", "vq-vae", "pca", "imitation_learning", "reinforcement_learning", "scripts")
SKIP_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}


def iter_python_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for child in root.rglob("*.py"):
        if any(part in SKIP_PARTS for part in child.parts):
            continue
        paths.append(child)
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    roots = [Path(arg) for arg in args] if args else [Path(root) for root in DEFAULT_ROOTS]
    failures: list[str] = []
    for root in roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = iter_python_files(root)
        for path in candidates:
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                failures.append(f"{path}:{exc.lineno}:{exc.offset}: {exc.msg}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Python syntax check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
