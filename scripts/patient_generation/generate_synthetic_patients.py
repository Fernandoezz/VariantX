import random
import pandas as pd


random.seed(42)  # reproducibility

N_PATIENTS = 5000          # real training target
MAX_PATIENTS_PER_GENE = 3  # gene diversity cap
N_BACKGROUND_VARIANTS = 100
N_HARD_NEGATIVES = 3        # guaranteed minimum per patient
SYMPTOM_KEEP_FRACTION = (0.6, 0.9)  # sample this fraction of true symptoms
N_NOISE_SYMPTOMS = (0, 2)


def load_data():
    snv = pd.read_csv("data/processed/snv/snv_features.csv", dtype={"chromosome": str}, low_memory=False)
    hpo = pd.read_csv("data/interim/hpo_tables/disease_hpo_annotations_clean.csv", low_memory=False)
    clinvar = pd.read_csv("data/interim/clinvar_tables/clinvar_variants_clean.csv", dtype={"chromosome": str})
    return snv, hpo, clinvar


def build_eligible_causal_pool(snv):
    """Pathogenic/Likely_pathogenic variants with a confirmed/unconfirmed Mendelian
    disease link - these can serve as the 'true answer' for a synthetic patient."""
    eligible = snv[
        snv["clinvar_significance"].isin(["Pathogenic", "Likely_pathogenic"]) &
        snv["disease_name"].notna() &
        snv["phenotype_type"].isin(["confirmed_mendelian", "unconfirmed_mendelian"])
    ]
    return eligible


def sample_causal_variants(eligible, n_patients, max_per_gene):
    """Sample causal variants with a per-gene cap to force gene/disease diversity."""
    genes = eligible["gene_symbol"].unique().tolist()
    random.shuffle(genes)

    selected_rows = []
    for gene in genes:
        if len(selected_rows) >= n_patients:
            break
        gene_rows = eligible[eligible["gene_symbol"] == gene]
        take_n = min(max_per_gene, len(gene_rows), n_patients - len(selected_rows))
        sampled = gene_rows.sample(n=take_n, random_state=random.randint(0, 999999))
        selected_rows.append(sampled)

    result = pd.concat(selected_rows, ignore_index=True)
    return result.iloc[:n_patients]


def build_hpo_lookup(hpo_df):
    """Precompute disease_id -> symptom DataFrame lookup ONCE, instead of
    filtering the full 282k-row table on every patient."""
    filtered = hpo_df[(hpo_df["aspect"] == "P") & (hpo_df["is_negated"] == False)]
    return {disease_id: group for disease_id, group in filtered.groupby("disease_id")}


def sample_patient_symptoms(disease_symptoms, hpo_df):
    """Sample a realistic, incomplete, slightly noisy symptom set for a patient."""
    if len(disease_symptoms) == 0:
        return []

    # Deduplicate in case the same disease-symptom pair appears more than once
    # (common in .hpoa files, sourced from multiple publications).
    disease_symptoms = disease_symptoms.drop_duplicates(subset=["hpo_id"]).reset_index(drop=True)

    keep_fraction = random.uniform(*SYMPTOM_KEEP_FRACTION)
    n_keep = max(1, int(len(disease_symptoms) * keep_fraction))
    n_sample = min(n_keep, len(disease_symptoms))

    weights = pd.to_numeric(disease_symptoms["frequency_numeric"], errors="coerce").fillna(0.3) + 0.01
    weights_list = weights.tolist()

    hpo_ids = disease_symptoms["hpo_id"].tolist()

    # random.sample doesn't support weights directly in older Python, so use
    # weighted selection without replacement via repeated random.choices + dedup
    chosen_indices = set()
    attempts = 0
    while len(chosen_indices) < n_sample and attempts < n_sample * 20:
        pick = random.choices(range(len(hpo_ids)), weights=weights_list, k=1)[0]
        chosen_indices.add(pick)
        attempts += 1

    patient_symptoms = [hpo_ids[i] for i in chosen_indices]

    n_noise = random.randint(*N_NOISE_SYMPTOMS)
    if n_noise > 0:
        random_symptoms = hpo_df[hpo_df["aspect"] == "P"].sample(n=n_noise, random_state=random.randint(0, 999999))
        patient_symptoms.extend(random_symptoms["hpo_id"].tolist())

    return list(set(patient_symptoms))


def assign_zygosity(inheritance_modes):
    """Assign a plausible zygosity based on the causal variant's inheritance mode."""
    if pd.isna(inheritance_modes):
        return random.choice(["heterozygous", "homozygous"])

    modes = inheritance_modes.lower()
    if "recessive" in modes:
        return "homozygous"
    if "dominant" in modes:
        return "heterozygous"
    if "x-linked" in modes:
        return random.choice(["hemizygous", "heterozygous"])
    return random.choice(["heterozygous", "homozygous"])


def simulate_quality_metadata():
    return {
        "simulated_read_depth": random.randint(15, 100),
        "simulated_genotype_quality": random.randint(20, 99),
        "simulated_filter_status": random.choices(["PASS", "LOWQUAL"], weights=[0.95, 0.05])[0],
    }


def sample_background_variants(general_pool, hard_negative_pool, causal_gene, n_background, n_hard_negatives):
    """Sample first, exclude the causal gene after - much faster than
    pre-filtering a multi-million-row pool on every single patient."""

    # Oversample slightly to leave room for removing any accidental causal-gene matches
    hard_negatives = hard_negative_pool.sample(n=min(n_hard_negatives + 2, len(hard_negative_pool)), random_state=random.randint(0, 999999))
    hard_negatives = hard_negatives[hard_negatives["gene_symbol"] != causal_gene].head(n_hard_negatives)

    remaining_needed = n_background - len(hard_negatives)
    general_background = general_pool.sample(n=min(remaining_needed + 5, len(general_pool)), random_state=random.randint(0, 999999))
    general_background = general_background[general_background["gene_symbol"] != causal_gene].head(remaining_needed)

    hard_negative_ids = set(hard_negatives["variant_id"])

    background_df = pd.concat([hard_negatives, general_background], ignore_index=True)
    background_df = background_df.drop_duplicates(subset=["variant_id"])

    return background_df, hard_negative_ids


def generate_patients(snv, hpo, clinvar, n_patients):
    eligible = build_eligible_causal_pool(snv)
    print(f"Eligible causal variant pool: {len(eligible)} rows, {eligible['gene_symbol'].nunique()} unique genes")

    # Precompute reusable pools ONCE - this is the actual speed fix
    print("Precomputing background variant pools...")
    general_pool = clinvar[clinvar["clinvar_significance"].isin(["Benign", "Likely_benign", "Uncertain_significance"])]
    hard_negative_pool = clinvar[clinvar["clinvar_significance"].isin(["Uncertain_significance", "Conflicting"])]

    print("Precomputing HPO symptom lookup...")
    hpo_lookup = build_hpo_lookup(hpo)
    all_p_symptoms = hpo[hpo["aspect"] == "P"]

    causal_sample = sample_causal_variants(eligible, int(n_patients * 1.2), MAX_PATIENTS_PER_GENE)
    print(f"Sampled {len(causal_sample)} candidate causal variants across {causal_sample['gene_symbol'].nunique()} unique genes")

    patients_meta = []
    patient_variants = []
    skipped_no_symptoms = 0

    patient_id = 0
    for _, causal_row in causal_sample.iterrows():
        if patient_id >= n_patients:
            break

        disease_id = f"OMIM:{int(causal_row['phenotype_mim_number'])}"
        disease_symptoms = hpo_lookup.get(disease_id, pd.DataFrame())

        patient_symptoms = sample_patient_symptoms(disease_symptoms, all_p_symptoms)

        if len(patient_symptoms) == 0:
            skipped_no_symptoms += 1
            continue

        patient_id += 1
        zygosity = assign_zygosity(causal_row["inheritance_modes"])

        patients_meta.append({
            "patient_id": patient_id,
            "causal_variant_id": causal_row["variant_id"],
            "causal_gene": causal_row["gene_symbol"],
            "causal_disease": causal_row["disease_name"],
            "causal_disease_mim": causal_row["phenotype_mim_number"],
            "causal_inheritance_mode": causal_row["inheritance_modes"],
            "n_patient_symptoms": len(patient_symptoms),
            "patient_hpo_terms": ";".join(patient_symptoms),
        })

        causal_quality = simulate_quality_metadata()
        patient_variants.append({
            "patient_id": patient_id,
            "variant_id": causal_row["variant_id"],
            "gene_symbol": causal_row["gene_symbol"],
            "is_causal": 1,
            "is_hard_negative": 0,
            "zygosity": zygosity,
            **causal_quality
        })

        background_df, hard_negative_ids = sample_background_variants(
            general_pool, hard_negative_pool, causal_row["gene_symbol"], N_BACKGROUND_VARIANTS, N_HARD_NEGATIVES
        )

        for _, bg_row in background_df.iterrows():
            bg_quality = simulate_quality_metadata()
            patient_variants.append({
                "patient_id": patient_id,
                "variant_id": bg_row["variant_id"],
                "gene_symbol": bg_row["gene_symbol"],
                "is_causal": 0,
                "is_hard_negative": 1 if bg_row["variant_id"] in hard_negative_ids else 0,
                "zygosity": random.choice(["heterozygous", "homozygous"]),
                **bg_quality
            })

        if patient_id % 500 == 0:
            print(f"  ...generated {patient_id} patients")

    print(f"Skipped candidates with no HPO symptom coverage: {skipped_no_symptoms}")
    if patient_id < n_patients:
        print(f"WARNING: only generated {patient_id}/{n_patients} patients")

    return pd.DataFrame(patients_meta), pd.DataFrame(patient_variants)


if __name__ == "__main__":
    snv, hpo, clinvar = load_data()

    patients_meta_df, patient_variants_df = generate_patients(snv, hpo, clinvar, N_PATIENTS)

    print()
    print(f"Total patients generated: {len(patients_meta_df)}")
    print(f"Total patient-variant rows: {len(patient_variants_df)}")
    print(f"Average variants per patient: {len(patient_variants_df) / len(patients_meta_df):.1f}")
    print(f"Patients with 0 symptoms sampled: {(patients_meta_df['n_patient_symptoms'] == 0).sum()}")
    print(f"Hard negatives total: {patient_variants_df['is_hard_negative'].sum()}")

    patients_meta_df.to_csv("data/processed/snv/patients_meta.csv", index=False)
    patient_variants_df.to_csv("data/processed/snv/patient_variants.csv", index=False)