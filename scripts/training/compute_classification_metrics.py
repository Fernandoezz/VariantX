import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def parse_multi_value_score(value):
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


def compute_classification_metrics_for_model(model_name, train_path, test_path,
                                               drop_columns, categorical_columns,
                                               multi_value_columns=None, threshold=0.5):
    print(f"\n{'='*20} {model_name.upper()} classification metrics {'='*20}")

    train_df = pd.read_csv(train_path, dtype={"chromosome": str}, low_memory=False)
    test_df = pd.read_csv(test_path, dtype={"chromosome": str}, low_memory=False)

    def prep(df):
        df = df.copy()
        if multi_value_columns:
            for col in multi_value_columns:
                if col in df.columns:
                    df[col] = df[col].apply(parse_multi_value_score)
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")
        keep = [c for c in df.columns if c not in drop_columns and c != "is_causal"]
        return df[keep]

    X_train = prep(train_df)
    y_train = train_df["is_causal"]
    X_test = prep(test_df)
    y_test = test_df["is_causal"]

    categorical_features = [c for c in categorical_columns if c in X_train.columns]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.05, "verbose": -1, "seed": 42,
    }
    model = lgb.train(params, train_data, num_boost_round=200)

    probabilities = model.predict(X_test)
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    print(f"Threshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"Confusion matrix -> TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"(Note: {y_test.sum()} true positives out of {len(y_test)} total rows - "
          f"heavily imbalanced by design, since each patient has exactly 1 causal variant "
          f"among ~{len(y_test)//y_test.sum()} candidates)")

    return {
        "model": model_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


if __name__ == "__main__":
    SNV_DROP = [
        "patient_id", "variant_id", "gene_symbol", "best_matching_disease",
        "chromosome", "position", "reference_allele", "alternate_allele",
        "is_hard_negative", "inheritance_modes"
    ]
    SNV_CATEGORICAL = [
        "clinvar_significance", "review_status", "zygosity", "gene_biotype",
        "haploinsufficiency_label", "SIFT_pred", "Polyphen2_HDIV_pred",
        "Polyphen2_HVAR_pred", "simulated_filter_status"
    ]
    SNV_MULTI_VALUE = ["SIFT_score", "Polyphen2_HDIV_score", "Polyphen2_HVAR_score"]

    snv_metrics = compute_classification_metrics_for_model(
        "snv",
        "data/processed/snv/snv_train.csv",
        "data/processed/snv/snv_test.csv",
        SNV_DROP, SNV_CATEGORICAL, SNV_MULTI_VALUE
    )

    CNV_DROP = [
        "patient_id", "variant_id", "gene_symbol", "best_matching_disease",
        "chromosome", "deletion_start", "deletion_end", "is_hard_negative"
    ]
    CNV_CATEGORICAL = ["zygosity", "gene_biotype", "haploinsufficiency_label", "simulated_filter_status"]

    cnv_metrics = compute_classification_metrics_for_model(
        "cnv",
        "data/processed/cnv/cnv_train.csv",
        "data/processed/cnv/cnv_test.csv",
        CNV_DROP, CNV_CATEGORICAL
    )

    summary = pd.DataFrame([snv_metrics, cnv_metrics])
    print("\n\n=== CLASSIFICATION METRICS SUMMARY ===")
    print(summary.to_string(index=False))

    summary.to_csv("data/processed/classification_metrics_summary.csv", index=False)