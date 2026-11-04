"""Locate the ACTG175 dataset.

Priority:
1. the ACTG175_PATH environment variable, if it points at a file;
2. <repo>/data/ACTG175.csv (the copy shipped with this repository);
3. the author's historical location, kept only as a fallback.
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def data_path() -> str:
    env = os.environ.get("ACTG175_PATH")
    if env and Path(env).exists():
        return str(Path(env))
    for candidate in (
        _HERE.parent / "data" / "ACTG175.csv",
        Path.home() / "Downloads" / "Machine-Learning-Project" / "data" / "ACTG175.csv",
    ):
        if candidate.exists():
            return str(candidate)
    return str(_HERE.parent / "data" / "ACTG175.csv")
