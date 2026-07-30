import pandas as pd
import numpy as np
import lightgbm as lgb
import shap


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


def prepare_features(df, drop_columns, categorical_columns, multi_value_columns=None):
    df = df.copy()

    if multi_value_columns:
        for col in multi_value_columns:
            if col in df.columns:
                df[col] = df[col].apply(parse_multi_value_score)

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    keep_columns = [c for c in df.columns if c not in drop_columns and c != "is_causal"]
    X = df[keep_columns]
    y = df["is_causal"]
    return X, y


def train_model(X_train, y_train, categorical_features):
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.05, "verbose": -1, "seed": 42,
    }
    return lgb.train(params, train_data, num_boost_round=200)


def run_shap_for_model(model_name, train_path, test_path, drop_columns, categorical_columns, multi_value_columns, meta_columns):
    print(f"\n{'='*20} SHAP analysis: {model_name} {'='*20}")

    train_df = pd.read_csv(train_path, dtype={"chromosome": str}, low_memory=False)
    test_df = pd.read_csv(test_path, dtype={"chromosome": str}, low_memory=False)

    X_train, y_train = prepare_features(train_df, drop_columns, categorical_columns, multi_value_columns)
    X_test, y_test = prepare_features(test_df, drop_columns, categorical_columns, multi_value_columns)

    categorical_features = [c for c in categorical_columns if c in X_train.columns]

    print("Training model...")
    model = train_model(X_train, y_train, categorical_features)

    print("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification LightGBM, shap_values may be a list [class0, class1]
    # or a single array depending on version - normalize to single array for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Global feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n--- Global Feature Importance (mean |SHAP value|) ---")
    print(importance_df.to_string(index=False))

    importance_df.to_csv(f"data/processed/{model_name}/shap_feature_importance.csv", index=False)

    # Per-prediction explanation for a few example causal variants
    print("\n--- Example per-patient explanations (causal variants only) ---")
    causal_mask = test_df["is_causal"] == 1
    example_indices = test_df[causal_mask].index[:3]

    explanations = []
    for idx in example_indices:
        pos_in_X = X_test.index.get_loc(idx)
        row_shap = shap_values[pos_in_X]
        row_features = X_test.iloc[pos_in_X]

        contribs = pd.DataFrame({
            "feature": X_test.columns,
            "value": row_features.values,
            "shap_value": row_shap
        }).sort_values("shap_value", key=abs, ascending=False)

        supporting = contribs[contribs["shap_value"] > 0].head(5)
        opposing = contribs[contribs["shap_value"] < 0].head(5)

        patient_id = test_df.loc[idx, "patient_id"]
        print(f"\nPatient {patient_id} (causal variant):")
        print("  Supporting evidence (pushed score UP):")
        for _, row in supporting.iterrows():
            print(f"    {row['feature']}: value={row['value']}, shap={row['shap_value']:.4f}")
        print("  Opposing evidence (pushed score DOWN):")
        for _, row in opposing.iterrows():
            print(f"    {row['feature']}: value={row['value']}, shap={row['shap_value']:.4f}")

        explanations.append({
            "patient_id": patient_id,
            "top_supporting": "; ".join(f"{r['feature']}={r['shap_value']:.3f}" for _, r in supporting.iterrows()),
            "top_opposing": "; ".join(f"{r['feature']}={r['shap_value']:.3f}" for _, r in opposing.iterrows()),
        })

    pd.DataFrame(explanations).to_csv(f"data/processed/{model_name}/shap_example_explanations.csv", index=False)

    return importance_df


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

    run_shap_for_model(
        "snv",
        "data/processed/snv/snv_train.csv",
        "data/processed/snv/snv_test.csv",
        SNV_DROP, SNV_CATEGORICAL, SNV_MULTI_VALUE, None
    )

    CNV_DROP = [
        "patient_id", "variant_id", "gene_symbol", "best_matching_disease",
        "chromosome", "deletion_start", "deletion_end", "is_hard_negative"
    ]
    CNV_CATEGORICAL = ["zygosity", "gene_biotype", "haploinsufficiency_label", "simulated_filter_status"]

    run_shap_for_model(
        "cnv",
        "data/processed/cnv/cnv_train.csv",
        "data/processed/cnv/cnv_test.csv",
        CNV_DROP, CNV_CATEGORICAL, None, None
    )