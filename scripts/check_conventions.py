"""Check the repository's small set of enforceable Python conventions."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_PUBLIC_INTERNAL_FIELDS = frozenset(
    {
        "lock",
        "pool",
        "saved_cpu",
        "saved_cuda",
        "store",
    }
)


def main() -> int:
    """Return a non-zero status when implementation state uses public names."""
    failures: list[str] = []
    for path in sorted(Path("kvfold").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in FORBIDDEN_PUBLIC_INTERNAL_FIELDS
                ):
                    failures.append(f"{path}:{target.lineno}: use self._{target.attr}")
    if failures:
        print("semi-private naming violations:")
        print("\n".join(failures))
        return 1
    print("semi-private naming: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
