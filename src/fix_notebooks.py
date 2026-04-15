from __future__ import annotations

import glob
from pathlib import Path

import nbformat


def fix_notebook(path: Path) -> bool:
    nb = nbformat.read(str(path), as_version=4)
    changed = False
    for cell in nb.cells:
        src = cell.get("source", "")
        if isinstance(src, str):
            new = src.replace("\\n\n", "\n")
            new = new.replace("\\n", "\n")
            if new.endswith("\\n"):
                new = new[:-2]
            if new != src:
                cell["source"] = new
                changed = True
    if changed:
        nbformat.write(nb, str(path))
    return changed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted(Path(p).resolve() for p in glob.glob(str(root / "notebooks" / "*.ipynb")))
    for p in paths:
        changed = fix_notebook(p)
        print(("fixed" if changed else "ok"), p.name)


if __name__ == "__main__":
    main()

