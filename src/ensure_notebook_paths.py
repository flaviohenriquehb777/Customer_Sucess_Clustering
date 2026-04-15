from __future__ import annotations

import glob
from pathlib import Path

import nbformat


BLOCK = [
    "import sys\n",
    "\n",
    "sys.path.append(str(Path('..').resolve()))\n",
    "\n",
]


def ensure_sys_path_in_notebook(path: Path) -> bool:
    nb = nbformat.read(str(path), as_version=4)
    changed = False

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if not isinstance(src, str) or not src.strip():
            continue

        if "sys.path.append" in src:
            break

        lines = src.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines[:10]):
            if line.strip() == "from pathlib import Path":
                insert_at = i + 1
                break

        lines[insert_at:insert_at] = BLOCK
        cell["source"] = "".join(lines)
        changed = True
        break

    if changed:
        nbformat.write(nb, str(path))
    return changed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted(Path(p).resolve() for p in glob.glob(str(root / "notebooks" / "*.ipynb")))
    for p in paths:
        changed = ensure_sys_path_in_notebook(p)
        print(("updated" if changed else "ok"), p.name)


if __name__ == "__main__":
    main()

