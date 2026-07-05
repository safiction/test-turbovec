import kagglehub
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "anomaly_detection"
TRAIN_SIZE = 200_000

def load_and_save():
    print("Loading kaggle dataset...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud") + "/creditcard.csv"
    path = path.replace('\\', "/")
    # path = 'C:/Users/Honor/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv'

    df = pd.read_csv(path)

    print(df.head())
    print(f"Total rows: {len(df)}")

    df = df.sort_values(by = "Time").reset_index(drop=True)

    X_all = df[[f"V{i}" for i in range(1, 29)]].to_numpy().astype(np.float32)
    y_all = df["Class"].to_numpy().astype(np.int64)

    X_train_raw = X_all[:TRAIN_SIZE]
    y_train_raw = y_all[:TRAIN_SIZE]

    X_train = X_train_raw[y_train_raw == 0]

    X_test = X_all[TRAIN_SIZE:]
    y_test = y_all[TRAIN_SIZE:]

    
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")


if __name__ == "__main__":
    load_and_save()