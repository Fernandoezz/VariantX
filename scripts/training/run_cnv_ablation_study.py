import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


BASE_DROP_COLUMNS = [
    "patient_id", "variant_id", "gene_symbol", "best_matching_disease",
    "chromosome", "deletion_start", "deletion_end", "is_hard_negative"
]

CATEGORICAL_COLUMNS = ["zygosity", "gene_biotype", "haploinsufficiency_label", "simulated_filter_status"]

LABEL_COLUMN = "is_causal"

GENE_LEVEL_COLUMNS = [
    "oe_lof", "oe_lof_lower", "oe_lof_upper", "oe_lof_upper_rank", "pLI",
    "haploinsufficiency_score", "triplosensitivity_score",
    "gene_start_position", "gene_end_position", "gene_biotype"
]
QUALITY_COLUMNS = ["simulated_read_depth", "simulated_genotype_quality", "simulated_filter_status"]
CNV_STRUCTURAL_COLUMNS = ["deletion_size", "allele_frequency", "overlap_length", "overlap_fraction_of_gene", "overlap_fraction_of_deletion"]

ABLATIONS = {
    "Full model": [],
    "A1: without phenotype similarity": ["phenotype_similarity_score"],
    "A2: without zygosity": ["zygosity"],
    "A3: without gene-level evidence": GENE_LEVEL_COLUMNS,
    "A4: without quality features": QUALITY_COLUMNS,
    "A5: without CNV structural features": CNV_STRUCTURAL_COLUMNS,
}


def prepare_features(df, extra_drop_columns):
    df = df.copy()

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
        "n_patients": n_patients,
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

    all_results = []

    for ablation_name, drop_cols in ABLATIONS.items():
        print(f"\n=== {ablation_name} ===")

        X_train, y_train, _ = prepare_features(train_df, drop_cols)
        X_test, y_test, test_patient_ids = prepare_features(test_df, drop_cols)
        categorical_features = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]

        model = train_model(X_train, y_train, categorical_features)
        metrics = evaluate_ranking(model, X_test, y_test, test_patient_ids)
        metrics["ablation"] = ablation_name
        metrics["n_features"] = len(X_train.columns)
        all_results.append(metrics)

        print(f"Top-1: {metrics['top1_accuracy']:.4f} | Top-5: {metrics['top5_accuracy']:.4f} | "
              f"MRR: {metrics['mrr']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df[["ablation", "n_features", "top1_accuracy", "top5_accuracy", "top10_accuracy", "mrr", "roc_auc"]]

    print("\n\n=== CNV ABLATION STUDY SUMMARY ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("data/processed/cnv/ablation_study_results.csv", index=False)