import pandas as pd
import numpy as np
import lightgbm as lgb


def rank_and_score(df, score_column, ascending=False):
    """Rank each patient's candidates by a single column, compute Top-1/5/10 and MRR."""
    top1_hits, top5_hits, top10_hits = 0, 0, 0
    reciprocal_ranks = []
    n_patients = 0

    for patient_id, group in df.groupby("patient_id"):
        group_sorted = group.sort_values(score_column, ascending=ascending, na_position="last").reset_index(drop=True)
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

    return {
        "n_patients": n_patients,
        "top1_accuracy": top1_hits / n_patients,
        "top5_accuracy": top5_hits / n_patients,
        "top10_accuracy": top10_hits / n_patients,
        "mrr": np.mean(reciprocal_ranks),
    }


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


def get_full_model_predictions(train_df, test_df):
    """Reuse the same full-feature LightGBM model to get its ranking scores
    on the test set, for direct comparison against the two baselines."""

    BASE_DROP_COLUMNS = [
        "patient_id", "variant_id", "gene_symbol", "best_matching_disease",
        "chromosome", "position", "reference_allele", "alternate_allele",
        "is_hard_negative", "inheritance_modes", "is_causal"
    ]
    CATEGORICAL_COLUMNS = [
        "clinvar_significance", "review_status", "zygosity", "gene_biotype",
        "haploinsufficiency_label", "SIFT_pred", "Polyphen2_HDIV_pred",
        "Polyphen2_HVAR_pred", "simulated_filter_status"
    ]
    MULTI_VALUE_SCORE_COLUMNS = ["SIFT_score", "Polyphen2_HDIV_score", "Polyphen2_HVAR_score"]

    def prep(df):
        df = df.copy()
        for col in MULTI_VALUE_SCORE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(parse_multi_value_score)
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        keep = [c for c in df.columns if c not in BASE_DROP_COLUMNS]
        return df[keep]

    X_train = prep(train_df)
    y_train = train_df["is_causal"]
    X_test = prep(test_df)

    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.05, "verbose": -1, "seed": 42,
    }
    model = lgb.train(params, train_data, num_boost_round=200)

    return model.predict(X_test)


if __name__ == "__main__":
    print("Loading data...")
    train_df = pd.read_csv("data/processed/snv/snv_train.csv", dtype={"chromosome": str}, low_memory=False)
    test_df = pd.read_csv("data/processed/snv/snv_test.csv", dtype={"chromosome": str}, low_memory=False)

    print("\n=== Full model (all features combined) ===")
    full_model_scores = get_full_model_predictions(train_df, test_df)

    test_df_baselines = test_df.copy()
    test_df_baselines["CADD_phred_numeric"] = pd.to_numeric(test_df_baselines["CADD_phred"], errors="coerce")
    test_df_baselines["full_model_score"] = full_model_scores

    print("\n=== Baseline 1: CADD-only ranking ===")
    cadd_metrics = rank_and_score(test_df_baselines, "CADD_phred_numeric", ascending=False)
    print(cadd_metrics)

    print("\n=== Baseline 2: Phenotype-only ranking ===")
    phenotype_metrics = rank_and_score(test_df_baselines, "phenotype_similarity_score", ascending=False)
    print(phenotype_metrics)

    print("\n=== Full model metrics ===")
    full_model_metrics = rank_and_score(test_df_baselines, "full_model_score", ascending=False)
    print(full_model_metrics)

    print("\n\n=== BASELINE COMPARISON SUMMARY ===")
    summary = pd.DataFrame([
        {"method": "CADD-only (pathogenicity baseline)", **cadd_metrics},
        {"method": "Phenotype-only (Exomiser-style baseline)", **phenotype_metrics},
        {"method": "Full model (VariantX)", **full_model_metrics},
    ])
    print(summary.to_string(index=False))

    summary.to_csv("data/processed/snv/baseline_comparison_results.csv", index=False)