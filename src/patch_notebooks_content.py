from __future__ import annotations

from pathlib import Path

import nbformat


REPLACEMENTS = [
    (
        "pd.DataFrame({'numeric': spec.numeric, 'categorical': spec.categorical}).to_json(PROCESSED_DIR / 'feature_spec.json', orient='records', force_ascii=False)",
        "pd.DataFrame({'tipo': ['numeric']*len(spec.numeric) + ['categorical']*len(spec.categorical), 'feature': spec.numeric + spec.categorical}).to_json(PROCESSED_DIR / 'feature_spec.json', orient='records', force_ascii=False)",
    ),
    (
        "    multi_class='multinomial',\n",
        "",
    ),
]


def patch_notebook(path: Path) -> bool:
    nb = nbformat.read(str(path), as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if not isinstance(src, str):
            continue
        new = src
        for old, repl in REPLACEMENTS:
            if old in new:
                new = new.replace(old, repl)
        if new != src:
            cell["source"] = new
            changed = True
    if changed:
        nbformat.write(nb, str(path))
    return changed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    for p in paths:
        changed = patch_notebook(p)
        print(("patched" if changed else "ok"), p.name)


if __name__ == "__main__":
    main()

