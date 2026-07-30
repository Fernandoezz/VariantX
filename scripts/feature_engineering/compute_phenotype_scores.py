import pandas as pd


def build_patient_symptom_sets(patients_meta):
    result = {}
    for _, row in patients_meta.iterrows():
        symptoms = row["patient_hpo_terms"].split(";") if pd.notna(row["patient_hpo_terms"]) else []
        result[row["patient_id"]] = set(symptoms)
    return result


def build_gene_to_diseases(omim_df):
    result = {}
    for gene, group in omim_df.dropna(subset=["phenotype_mim_number"]).groupby("gene_symbol"):
        disease_ids = [f"OMIM:{int(mim)}" for mim in group["phenotype_mim_number"].unique()]
        result[gene] = disease_ids
    return result


def build_disease_symptom_freq(hpo_df):
    filtered = hpo_df[(hpo_df["aspect"] == "P") & (hpo_df["is_negated"] == False)].copy()
    filtered["frequency_numeric"] = pd.to_numeric(filtered["frequency_numeric"], errors="coerce").fillna(0.3)

    result = {}
    for disease_id, group in filtered.groupby("disease_id"):
        result[disease_id] = dict(zip(group["hpo_id"], group["frequency_numeric"]))
    return result


def score_patient_against_disease(patient_symptoms, disease_symptom_freq):
    if not disease_symptom_freq:
        return 0.0

    total_weight = sum(disease_symptom_freq.values())
    if total_weight == 0:
        return 0.0

    matched_weight = sum(
        freq for hpo_id, freq in disease_symptom_freq.items() if hpo_id in patient_symptoms
    )
    return matched_weight / total_weight


def compute_scores(patient_variants, patient_symptom_sets, gene_to_diseases, disease_symptom_freq):
    scores = []
    best_diseases = []

    for row in patient_variants.itertuples():
        patient_symptoms = patient_symptom_sets.get(row.patient_id, set())
        candidate_diseases = gene_to_diseases.get(row.gene_symbol, [])

        if not candidate_diseases or not patient_symptoms:
            scores.append(0.0)
            best_diseases.append(None)
            continue

        best_score = 0.0
        best_disease = None
        for disease_id in candidate_diseases:
            disease_freq = disease_symptom_freq.get(disease_id, {})
            score = score_patient_against_disease(patient_symptoms, disease_freq)
            if score > best_score:
                best_score = score
                best_disease = disease_id

        scores.append(best_score)
        best_diseases.append(best_disease)

    return scores, best_diseases


def add_phenotype_scores(patients_meta_path, patient_variants_path, omim_path, hpo_path, output_path):
    patients_meta = pd.read_csv(patients_meta_path)
    patient_variants = pd.read_csv(patient_variants_path)
    omim_df = pd.read_csv(omim_path)
    hpo_df = pd.read_csv(hpo_path, low_memory=False)

    print("Precomputing lookups...")
    patient_symptom_sets = build_patient_symptom_sets(patients_meta)
    gene_to_diseases = build_gene_to_diseases(omim_df)
    disease_symptom_freq = build_disease_symptom_freq(hpo_df)

    print(f"Scoring {len(patient_variants)} patient-variant rows...")
    scores, best_diseases = compute_scores(
        patient_variants, patient_symptom_sets, gene_to_diseases, disease_symptom_freq
    )

    patient_variants["phenotype_similarity_score"] = scores
    patient_variants["best_matching_disease"] = best_diseases

    print(f"Score distribution:")
    print(patient_variants["phenotype_similarity_score"].describe())
    print()
    print("Mean score for causal variants:", patient_variants[patient_variants["is_causal"] == 1]["phenotype_similarity_score"].mean())
    print("Mean score for background variants:", patient_variants[patient_variants["is_causal"] == 0]["phenotype_similarity_score"].mean())

    patient_variants.to_csv(output_path, index=False)
    return patient_variants


if __name__ == "__main__":
    print("=== SNV phenotype scoring ===")
    add_phenotype_scores(
        "data/processed/snv/patients_meta.csv",
        "data/processed/snv/patient_variants.csv",
        "data/interim/omim_tables/gene_disease_inheritance.csv",
        "data/interim/hpo_tables/disease_hpo_annotations_clean.csv",
        "data/processed/snv/patient_variants_scored.csv"
    )

    print()
    print("=== CNV phenotype scoring ===")
    add_phenotype_scores(
        "data/processed/cnv/patients_meta.csv",
        "data/processed/cnv/patient_variants.csv",
        "data/interim/omim_tables/gene_disease_inheritance.csv",
        "data/interim/hpo_tables/disease_hpo_annotations_clean.csv",
        "data/processed/cnv/patient_variants_scored.csv"
    )