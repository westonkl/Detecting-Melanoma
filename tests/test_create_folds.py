import numpy as np
import pandas as pd
from sklearn import model_selection


def generate_stratified_folds(
    df: pd.DataFrame, n_splits: int = 10, target_col: str = "target"
) -> pd.DataFrame:
    """Helper mirroring create_folds.py logic for testing."""
    df = df.copy()
    df["kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    y = df[target_col].values
    kf = model_selection.StratifiedKFold(n_splits=n_splits)

    for fold_, (_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = fold_

    return df


def test_stratified_kfold_structure():
    # Create synthetic dataset with 100 samples and imbalanced classes (80 zeros, 20 ones)
    targets = np.array([0] * 80 + [1] * 20)
    image_names = [f"ISIC_{i:07d}" for i in range(100)]
    df = pd.DataFrame({"image_name": image_names, "target": targets})

    df_folds = generate_stratified_folds(df, n_splits=10)

    # Verify no unassigned folds
    assert (df_folds["kfold"] >= 0).all()
    assert (df_folds["kfold"] < 10).all()
    assert len(df_folds["kfold"].unique()) == 10

    # Verify fold distribution
    for fold in range(10):
        val_subset = df_folds[df_folds.kfold == fold]
        # Each fold should have 10 samples (100 / 10)
        assert len(val_subset) == 10
        # Each fold should have exactly 2 positive targets (20 / 10) and 8 negative (80 / 10)
        assert (val_subset.target == 1).sum() == 2
        assert (val_subset.target == 0).sum() == 8


def test_stratified_kfold_all_rows_preserved():
    targets = np.random.choice([0, 1], size=200, p=[0.85, 0.15])
    df = pd.DataFrame(
        {"image_name": [f"img_{i}" for i in range(200)], "target": targets}
    )

    df_folds = generate_stratified_folds(df, n_splits=5)
    assert len(df_folds) == len(df)
    assert set(df_folds.columns) == {"image_name", "target", "kfold"}
    assert (
        df_folds.target.value_counts().to_dict() == df.target.value_counts().to_dict()
    )
