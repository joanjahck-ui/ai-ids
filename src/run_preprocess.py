"""Small runner to demonstrate preprocessing with synthetic data.

Run this from the project root (PowerShell):

    python .\src\run_preprocess.py

It will generate synthetic data, run preprocessing, show shapes, and save a preprocessor under `models/`.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so "src" is importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Check for required runtime packages before importing heavy modules
from src.requirement_checks import check_requirements, format_install_instructions
missing = check_requirements(["numpy", "pandas", "sklearn", "imbalanced_learn"])
if missing:
    print(format_install_instructions(missing))
    raise SystemExit(1)

from src.data_loader import generate_synthetic_cic_data, preprocess, stratified_split, save_preprocessor


def main():
    print("Generating synthetic data (small sample)...")
    df = generate_synthetic_cic_data(n_samples=1000)
    print("Running preprocessing (one-hot encode + z-score scaling)")
    out = preprocess(df, categorical_cols=["protocol", "service"], apply_smote=False)
    splits = stratified_split(out["X"], out["y"], test_size=0.15, val_size=0.15)
    print(f"Train: {splits['X_train'].shape}, Val: {splits['X_val'].shape}, Test: {splits['X_test'].shape}")
    preproc_path = save_preprocessor(out["scaler"], out["feature_columns"])
    print(f"Preprocessor saved to: {preproc_path}")


if __name__ == "__main__":
    main()
