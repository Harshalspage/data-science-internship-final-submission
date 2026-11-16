
import os
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_DIR = os.path.join(
    BASE_DIR, "data", "sample", "liar_dataset"
)

FILES = ["train.tsv", "valid.tsv", "test.tsv"]

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state", "party",
    "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts",
    "pants_fire_counts", "context"
]

for filename in FILES:
    filepath = os.path.join(DATA_DIR, filename)

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=COLUMNS
    )

    print("\n" + "=" * 55)
    print(f"FILE: {filename}")
    print("=" * 55)

    print("Dataset shape:", df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nMissing values:")
    print(df[["label", "statement"]].isnull().sum())

    print("\nSample statements:")
    print(
        df[["label", "statement"]]
        .head(3)
        .to_string(index=False)
    )
