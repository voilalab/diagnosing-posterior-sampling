"""Copy executed example notebooks into the Sphinx source tree."""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "examples"
DST = HERE / "source" / "_examples"


def main() -> None:
    """Mirror ``examples/*.ipynb`` into ``docsrc/source/_examples/``."""
    DST.mkdir(parents=True, exist_ok=True)
    for nb in SRC.glob("*.ipynb"):
        shutil.copy(nb, DST / nb.name)


if __name__ == "__main__":
    main()
