
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

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state", "party",
    "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts",
    "pants_fire_counts", "context"
]

FILES = {
    "train": "train.tsv",
    "valid": "valid.tsv",
    "test": "test.tsv"
}

LABEL_MAPPING = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 0,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0
}

for split, filename in FILES.items():

    filepath = os.path.join(DATA_DIR, filename)

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=COLUMNS
    )

    # Keep required columns
    df = df[["id", "label", "statement"]].copy()

    # Normalize text
    df["statement"] = (
        df["statement"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Map original labels
    df["binary_label"] = df["label"].map(LABEL_MAPPING)

    # Remove invalid or empty records
    df = df.dropna(subset=["binary_label"])
    df = df[df["statement"] != ""]

    df["binary_label"] = df["binary_label"].astype(int)

    # Save processed split
    output_path = os.path.join(
        OUTPUT_DIR,
        f"misinformation_{split}.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\n{split.upper()} DATASET")
    print("Records:", len(df))
    print("Class distribution:")
    print(df["binary_label"].value_counts().sort_index())
    print("Saved:", output_path)

print("\nDataset preparation completed!")
