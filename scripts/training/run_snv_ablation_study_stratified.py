import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


BASE_DROP_COLUMNS = [
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
MULTI_VALUE_SCORE_COLUMNS = ["SIFT_score", "Polyphen2_HDIV_score", "Polyphen2_HVAR_score"]

GENE_LEVEL_COLUMNS = [
    "oe_lof", "oe_lof_lower", "oe_lof_upper", "oe_lof_upper_rank", "pLI",
    "haploinsufficiency_score", "triplosensitivity_score", "haploinsufficiency_label",
    "gene_start_position", "gene_end_position", "gene_biotype"
]
QUALITY_COLUMNS = ["simulated_read_depth", "simulated_genotype_quality", "simulated_filter_status"]
VARIANT_FUNCTIONAL_COLUMNS = [
    "SIFT_score", "SIFT_converted_rankscore", "SIFT_pred",
    "Polyphen2_HDIV_score", "Polyphen2_HDIV_rankscore", "Polyphen2_HDIV_pred",
    "Polyphen2_HVAR_score", "Polyphen2_HVAR_rankscore", "Polyphen2_HVAR_pred",
    "CADD_raw", "CADD_raw_rankscore", "CADD_phred",
    "GERP_NR", "GERP_RS",
    "phyloP100way_vertebrate", "phyloP100way_vertebrate_rankscore",
    "phastCons100way_vertebrate", "phastCons100way_vertebrate_rankscore"
]

ABLATIONS = {
    "Full model": [],
    "A1: without phenotype similarity": ["phenotype_similarity_score"],
    "A2: without zygosity": ["zygosity"],
    "A3: without gene-level evidence": GENE_LEVEL_COLUMNS,
    "A4: without quality features": QUALITY_COLUMNS,
    "A5: without variant functional scores": VARIANT_FUNCTIONAL_COLUMNS,
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


def prepare_features(df, extra_drop_columns):
    df = df.copy()
    for col in MULTI_VALUE_SCORE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_multi_value_score)
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and col not in extra_drop_columns:
            df[col] = df[col].astype("category")

    all_drop = set(BASE_DROP_COLUMNS) | set(extra_drop_columns)
    keep_columns = [c for c in df.columns if c not in all_drop and c != LABEL_COLUMN]

    X = df[keep_columns]
    y = df[LABEL_COLUMN]
    patient_ids = df["patient_id"]
    return X, y, patient_ids


def train_model(X_train, y_train, categorical_features):
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.05, "verbose": -1, "seed": 42,
    }
    return lgb.train(params, train_data, num_boost_round=200)


def evaluate_ranking_stratified(model, X_test, y_test, patient_ids, tier_lookup):
    """Same ranking evaluation as before, but also breaks results down by
    each patient's phenotype_tier."""
    predictions = model.predict(X_test)

    results_df = pd.DataFrame({
        "patient_id": patient_ids.values,
        "is_causal": y_test.values,
        "predicted_score": predictions,
    })
    results_df["phenotype_tier"] = results_df["patient_id"].map(tier_lookup)

    def compute_metrics(df_subset):
        top1_hits, top5_hits, top10_hits = 0, 0, 0
        reciprocal_ranks = []
        n_patients = 0

        for patient_id, group in df_subset.groupby("patient_id"):
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

        if n_patients == 0:
            return None

        return {
            "n_patients": n_patients,
            "top1_accuracy": top1_hits / n_patients,
            "top5_accuracy": top5_hits / n_patients,
            "top10_accuracy": top10_hits / n_patients,
            "mrr": np.mean(reciprocal_ranks),
        }

    overall_metrics = compute_metrics(results_df)
    overall_metrics["roc_auc"] = roc_auc_score(y_test, predictions)
    overall_metrics["tier"] = "ALL"

    tier_results = [overall_metrics]
    for tier in results_df["phenotype_tier"].dropna().unique():
        subset = results_df[results_df["phenotype_tier"] == tier]
        tier_metrics = compute_metrics(subset)
        if tier_metrics:
            tier_metrics["tier"] = tier
            tier_metrics["roc_auc"] = None
            tier_results.append(tier_metrics)

    return tier_results


if __name__ == "__main__":
    print("Loading train/test data...")
    train_df = pd.read_csv("data/processed/snv/snv_train.csv", dtype={"chromosome": str}, low_memory=False)
    test_df = pd.read_csv("data/processed/snv/snv_test.csv", dtype={"chromosome": str}, low_memory=False)

    patients_meta = pd.read_csv("data/processed/snv/patients_meta.csv")
    tier_lookup = dict(zip(patients_meta["patient_id"], patients_meta["phenotype_tier"]))

    all_rows = []

    for ablation_name, drop_cols in ABLATIONS.items():
        print(f"\n=== {ablation_name} ===")

        X_train, y_train, _ = prepare_features(train_df, drop_cols)
        X_test, y_test, test_patient_ids = prepare_features(test_df, drop_cols)
        categorical_features = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]

        model = train_model(X_train, y_train, categorical_features)
        tier_results = evaluate_ranking_stratified(model, X_test, y_test, test_patient_ids, tier_lookup)

        for tr in tier_results:
            tr["ablation"] = ablation_name
            all_rows.append(tr)
            print(f"  [{tr['tier']:>22}] n={tr['n_patients']:>4} | Top-1: {tr['top1_accuracy']:.4f} | "
                  f"Top-5: {tr['top5_accuracy']:.4f} | MRR: {tr['mrr']:.4f}")

    results_df = pd.DataFrame(all_rows)
    results_df = results_df[["ablation", "tier", "n_patients", "top1_accuracy", "top5_accuracy", "top10_accuracy", "mrr", "roc_auc"]]

    print("\n\n=== STRATIFIED ABLATION SUMMARY ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("data/processed/snv/ablation_study_stratified_results.csv", index=False)