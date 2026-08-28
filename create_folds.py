import os
from pathlib import Path

import pandas as pd
from sklearn import model_selection

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = Path(os.getenv("MELANOMA_INPUT_PATH", BASE_DIR / "input"))

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH / "train.csv")
    df["kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffling dataframe
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=10)

    for fold_, (_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = fold_

    print(df.target.value_counts())

    df.to_csv(INPUT_PATH / "train_folds.csv", index=False)
