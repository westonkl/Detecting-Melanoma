import os
import pandas as pd

if __name__ == "__main__":
    # input_path = "E:\Users\Weston\workspace\Detecting-Melanoma\train.csv"
    input_path = "E:/Users/Weston/workspace/Detecting-Melanoma/"
    df = pd.read_csv(os.path.join(input_path, "train.csv"))
    df["kfold"] = -1