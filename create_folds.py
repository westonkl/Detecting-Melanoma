import os
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # input_path = "E:\Users\Weston\workspace\Detecting-Melanoma\train.csv"
    input_path = "E:/Users/Weston/workspace/Detecting-Melanoma/"
    df = pd.read_csv(os.path.join(input_path, "train.csv"))
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True) #shuffling dataframe
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=10)

    for fold_, (, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = fold_

    print(df.target.value_counts())