"""Simple runtime dependency checker.

Use this to provide a clear message when required packages are missing instead of
crashing with ModuleNotFoundError deep inside imports.
"""
from __future__ import annotations

import importlib
from typing import List


def check_requirements(packages: List[str]) -> List[str]:
    """Return a list of missing packages (import names).

    Example packages: ['numpy', 'pandas', 'sklearn', 'imbalanced_learn']
    """
    missing = []
    for pkg in packages:
        try:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)
        except Exception:
            # fallback to try/except import
            try:
                importlib.import_module(pkg)
            except Exception:
                missing.append(pkg)
    return missing


def format_install_instructions(missing: List[str]) -> str:
    if not missing:
        return "All required packages appear to be installed."
    pkgs = " ".join(missing)
    instructions = (
        "The following Python packages are missing: " + pkgs + "\n"
        "Install them in your active virtual environment with:\n"
        "pip install -r requirements.txt\n\n"
        "Or install only the missing ones:\n"
        f"pip install {' '.join(missing)}\n"
    )
    return instructions


if __name__ == "__main__":
    core = ["numpy", "pandas", "sklearn", "imbalanced_learn", "xgboost"]
    miss = check_requirements(core)
    print(format_install_instructions(miss))
