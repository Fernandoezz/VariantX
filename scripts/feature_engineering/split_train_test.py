import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_patient(training_df, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    train_idx, test_idx = next(
        splitter.split(training_df, groups=training_df["patient_id"])
    )

    train_df = training_df.iloc[train_idx].reset_index(drop=True)
    test_df = training_df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df


def verify_no_leakage(train_df, test_df):
    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    overlap = train_patients & test_patients
    return overlap


if __name__ == "__main__":
    print("=== SNV split ===")
    snv_df = pd.read_csv("data/processed/snv/snv_training_table_final.csv", dtype={"chromosome": str}, low_memory=False)

    snv_train, snv_test = split_by_patient(snv_df)

    print(f"Total rows: {len(snv_df)}")
    print(f"Train rows: {len(snv_train)} ({snv_train['patient_id'].nunique()} patients)")
    print(f"Test rows: {len(snv_test)} ({snv_test['patient_id'].nunique()} patients)")

    overlap = verify_no_leakage(snv_train, snv_test)
    print(f"Patient overlap between train/test (must be 0): {len(overlap)}")

    print(f"Causal variants in train: {snv_train['is_causal'].sum()}")
    print(f"Causal variants in test: {snv_test['is_causal'].sum()}")

    snv_train.to_csv("data/processed/snv/snv_train.csv", index=False)
    snv_test.to_csv("data/processed/snv/snv_test.csv", index=False)

    print("\n=== CNV split ===")
    cnv_df = pd.read_csv("data/processed/cnv/cnv_training_table.csv", dtype={"chromosome": str}, low_memory=False)

    cnv_train, cnv_test = split_by_patient(cnv_df)

    print(f"Total rows: {len(cnv_df)}")
    print(f"Train rows: {len(cnv_train)} ({cnv_train['patient_id'].nunique()} patients)")
    print(f"Test rows: {len(cnv_test)} ({cnv_test['patient_id'].nunique()} patients)")

    overlap = verify_no_leakage(cnv_train, cnv_test)
    print(f"Patient overlap between train/test (must be 0): {len(overlap)}")

    print(f"Causal variants in train: {cnv_train['is_causal'].sum()}")
    print(f"Causal variants in test: {cnv_test['is_causal'].sum()}")

    cnv_train.to_csv("data/processed/cnv/cnv_train.csv", index=False)
    cnv_test.to_csv("data/processed/cnv/cnv_test.csv", index=False)