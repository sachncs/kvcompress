"""Validate relative Markdown links in repository documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "#", "<")


def main() -> int:
    """Return non-zero when a relative Markdown target does not exist."""
    root = Path(__file__).resolve().parent.parent
    failures: list[str] = []
    for path in sorted(root.rglob("*.md")):
        if any(part in {".venv", "node_modules", "site/dist"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"{path.relative_to(root)} -> {raw_target}")
    if failures:
        print("Broken documentation links:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("Documentation links: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
