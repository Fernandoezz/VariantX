import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


DROP_COLUMNS = [
    "patient_id", "variant_id", "gene_symbol", "best_matching_disease",
    "chromosome", "position", "reference_allele", "alternate_allele",
    "is_hard_negative", "inheritance_modes"
]

CATEGORICAL_COLUMNS = [
    "clinvar_significance", "review_status", "zygosity", "gene_biotype",
    "haploinsufficiency_label", "SIFT_pred", "Polyphen2_HDIV_pred",
    "Polyphen2_HVAR_pred", "simulated_filter_status"
]

LABEL_COLUMN = "is_causal"


def parse_multi_value_score(value):
    """dbNSFP sometimes reports multiple scores per variant (one per
    overlapping transcript), semicolon-separated. Take the minimum
    (most damaging) as a single representative value."""
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except ValueError:
        parts = str(value).split(";")
        numeric_parts = []
        for p in parts:
            try:
                numeric_parts.append(float(p))
            except ValueError:
                continue
        return min(numeric_parts) if numeric_parts else np.nan


MULTI_VALUE_SCORE_COLUMNS = ["SIFT_score", "Polyphen2_HDIV_score", "Polyphen2_HVAR_score"]


def prepare_features(df):
    df = df.copy()

    for col in MULTI_VALUE_SCORE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_multi_value_score)

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
    """Per-patient ranking evaluation: Top-1, Top-5, Top-10 accuracy, MRR, NDCG."""
    predictions = model.predict(X_test)

    results_df = pd.DataFrame({
        "patient_id": patient_ids.values,
        "is_causal": y_test.values,
        "predicted_score": predictions,
    })

    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    reciprocal_ranks = []
    n_patients = 0

    for patient_id, group in results_df.groupby("patient_id"):
        group_sorted = group.sort_values("predicted_score", ascending=False).reset_index(drop=True)
        causal_rank_positions = group_sorted.index[group_sorted["is_causal"] == 1].tolist()

        if not causal_rank_positions:
            continue

        n_patients += 1
        rank = causal_rank_positions[0] + 1  # 1-indexed rank

        if rank == 1:
            top1_hits += 1
        if rank <= 5:
            top5_hits += 1
        if rank <= 10:
            top10_hits += 1

        reciprocal_ranks.append(1.0 / rank)

    top1_accuracy = top1_hits / n_patients
    top5_accuracy = top5_hits / n_patients
    top10_accuracy = top10_hits / n_patients
    mrr = np.mean(reciprocal_ranks)

    auc = roc_auc_score(y_test, predictions)

    return {
        "n_patients_evaluated": n_patients,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
        "mrr": mrr,
        "roc_auc": auc,
    }


if __name__ == "__main__":
    print("Loading train/test data...")
    train_df = pd.read_csv("data/processed/snv/snv_train.csv", dtype={"chromosome": str}, low_memory=False)
    test_df = pd.read_csv("data/processed/snv/snv_test.csv", dtype={"chromosome": str}, low_memory=False)

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

    print("\n=== SNV Model Evaluation ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    model.save_model("data/processed/snv/snv_model.txt")
    print("\nModel saved to data/processed/snv/snv_model.txt")