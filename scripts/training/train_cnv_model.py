import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


DROP_COLUMNS = [
    "patient_id", "variant_id", "gene_symbol", "best_matching_disease",
    "chromosome", "deletion_start", "deletion_end",
    "is_hard_negative"
]

CATEGORICAL_COLUMNS = [
    "zygosity", "gene_biotype", "haploinsufficiency_label", "simulated_filter_status"
]

LABEL_COLUMN = "is_causal"


def prepare_features(df):
    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    keep_columns = [c for c in df.columns if c not in DROP_COLUMNS and c != LABEL_COLUMN]
    X = df[keep_columns]
    y = df[LABEL_COLUMN]
    patient_ids = df["patient_id"]

    return X, y, patient_ids


def train_model(X_train, y_train, categorical_features):
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(params, train_data, num_boost_round=200)
    return model


def evaluate_ranking(model, X_test, y_test, patient_ids):
    predictions = model.predict(X_test)

    results_df = pd.DataFrame({
        "patient_id": patient_ids.values,
        "is_causal": y_test.values,
        "predicted_score": predictions,
    })

    top1_hits, top5_hits, top10_hits = 0, 0, 0
    reciprocal_ranks = []
    n_patients = 0

    for patient_id, group in results_df.groupby("patient_id"):
        group_sorted = group.sort_values("predicted_score", ascending=False).reset_index(drop=True)
        causal_positions = group_sorted.index[group_sorted["is_causal"] == 1].tolist()

        if not causal_positions:
            continue

        n_patients += 1
        rank = causal_positions[0] + 1

        if rank == 1:
            top1_hits += 1
        if rank <= 5:
            top5_hits += 1
        if rank <= 10:
            top10_hits += 1

        reciprocal_ranks.append(1.0 / rank)

    auc = roc_auc_score(y_test, predictions)

    return {
        "n_patients_evaluated": n_patients,
        "top1_accuracy": top1_hits / n_patients,
        "top5_accuracy": top5_hits / n_patients,
        "top10_accuracy": top10_hits / n_patients,
        "mrr": np.mean(reciprocal_ranks),
        "roc_auc": auc,
    }


if __name__ == "__main__":
    print("Loading train/test data...")
    train_df = pd.read_csv("data/processed/cnv/cnv_train.csv", dtype={"chromosome": str}, low_memory=False)
    test_df = pd.read_csv("data/processed/cnv/cnv_test.csv", dtype={"chromosome": str}, low_memory=False)

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    X_train, y_train, _ = prepare_features(train_df)
    X_test, y_test, test_patient_ids = prepare_features(test_df)

    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]

    print(f"Feature columns ({len(X_train.columns)}): {X_train.columns.tolist()}")
    print(f"Categorical features: {categorical_features}")
    print(f"Positive class (is_causal=1) in train: {y_train.sum()} / {len(y_train)}")

    print("\nTraining LightGBM model...")
    model = train_model(X_train, y_train, categorical_features)

    print("\nEvaluating on test set...")
    metrics = evaluate_ranking(model, X_test, y_test, test_patient_ids)

    print("\n=== CNV Model Evaluation ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    model.save_model("data/processed/cnv/cnv_model.txt")
    print("\nModel saved to data/processed/cnv/cnv_model.txt")