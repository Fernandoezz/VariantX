import random
import pandas as pd


random.seed(42)  # reproducibility

N_PATIENTS = 5000          # real training target 
MAX_PATIENTS_PER_GENE = 3  # gene diversity cap
N_BACKGROUND_VARIANTS = 100
N_HARD_NEGATIVES_UNCERTAIN = 3   # uncertain/conflicting significance, unrelated gene
N_HARD_NEGATIVES_PATHOGENIC = 2  # genuinely pathogenic, but in an UNRELATED gene
MIN_SHARED_SYMPTOMS_FOR_PHENOCOPY = 2

PHENOTYPE_TIERS = {
    "well_phenotyped":       {"probability": 0.3, "keep_fraction": (0.7, 0.9), "noise_symptoms": (0, 1)},
    "moderately_phenotyped": {"probability": 0.4, "keep_fraction": (0.4, 0.7), "noise_symptoms": (1, 3)},
    "poorly_phenotyped":     {"probability": 0.3, "keep_fraction": (0.15, 0.4), "noise_symptoms": (2, 5)},
}


def load_data():
    snv = pd.read_csv("data/processed/snv/snv_features.csv", dtype={"chromosome": str}, low_memory=False)
    hpo = pd.read_csv("data/interim/hpo_tables/disease_hpo_annotations_clean.csv", low_memory=False)
    clinvar = pd.read_csv("data/interim/clinvar_tables/clinvar_variants_clean.csv", dtype={"chromosome": str})
    omim_df = pd.read_csv("data/interim/omim_tables/gene_disease_inheritance.csv")
    return snv, hpo, clinvar, omim_df


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
    total = 0
    for gene in genes:
        if total >= n_patients:
            break
        gene_rows = eligible[eligible["gene_symbol"] == gene]
        take_n = min(max_per_gene, len(gene_rows), n_patients - total)
        sampled = gene_rows.sample(n=take_n, random_state=random.randint(0, 999999))
        selected_rows.append(sampled)
        total += take_n

    result = pd.concat(selected_rows, ignore_index=True)
    return result.iloc[:n_patients]


def build_hpo_lookup(hpo_df):
    """Precompute disease_id -> symptom DataFrame lookup ONCE, instead of
    filtering the full 282k-row table on every patient."""
    filtered = hpo_df[(hpo_df["aspect"] == "P") & (hpo_df["is_negated"] == False)]
    return {disease_id: group for disease_id, group in filtered.groupby("disease_id")}


def build_disease_symptom_sets(hpo_df):
    """disease_id -> set of HPO IDs (aspect=P, non-negated only)."""
    filtered = hpo_df[(hpo_df["aspect"] == "P") & (hpo_df["is_negated"] == False)]
    return {
        disease_id: set(group["hpo_id"])
        for disease_id, group in filtered.groupby("disease_id")
    }


def build_symptom_to_diseases_index(disease_symptom_sets):
    """hpo_id -> set of disease_ids that include it. Used to quickly find
    candidate phenocopy diseases without comparing every disease pair."""
    index = {}
    for disease_id, symptoms in disease_symptom_sets.items():
        for hpo_id in symptoms:
            index.setdefault(hpo_id, set()).add(disease_id)
    return index


def build_disease_to_gene_lookup(omim_df):
    """disease_id (OMIM:xxxxx) -> list of gene_symbols associated with it."""
    result = {}
    for _, row in omim_df.dropna(subset=["phenotype_mim_number"]).iterrows():
        disease_id = f"OMIM:{int(row['phenotype_mim_number'])}"
        result.setdefault(disease_id, []).append(row["gene_symbol"])
    return result


def build_gene_to_variants_lookup(clinvar_df):
    """gene_symbol -> DataFrame of its ClinVar variants. Precomputed ONCE
    so phenocopy variant selection never scans the full ClinVar table
    inside the per-patient loop."""
    return {gene: group for gene, group in clinvar_df.groupby("gene_symbol")}


def build_gene_to_inheritance_lookup(omim_df):
    """gene_symbol -> most common inheritance mode text (first non-null found).
    Used to assign background variants zygosity that's INCONSISTENT with
    their own gene's real inheritance requirement, creating a genuine,
    learnable inheritance-mismatch signal instead of pure random noise."""
    lookup = {}
    for _, row in omim_df.dropna(subset=["inheritance_modes"]).iterrows():
        gene = row["gene_symbol"]
        if gene not in lookup:
            lookup[gene] = row["inheritance_modes"]
    return lookup


def find_phenocopy_disease(causal_disease_id, disease_symptom_sets, symptom_to_diseases,
                             min_shared_symptoms=MIN_SHARED_SYMPTOMS_FOR_PHENOCOPY):
    """Find a different disease that shares at least min_shared_symptoms
    symptoms with the causal disease - a realistic differential diagnosis,
    not just a randomly unrelated disease."""
    causal_symptoms = disease_symptom_sets.get(causal_disease_id, set())
    if not causal_symptoms:
        return None

    candidate_counts = {}
    for hpo_id in causal_symptoms:
        for other_disease in symptom_to_diseases.get(hpo_id, set()):
            if other_disease == causal_disease_id:
                continue
            candidate_counts[other_disease] = candidate_counts.get(other_disease, 0) + 1

    qualifying = [d for d, count in candidate_counts.items() if count >= min_shared_symptoms]
    if not qualifying:
        return None

    return random.choice(qualifying)


def find_phenocopy_variant(causal_disease_id, causal_gene, disease_symptom_sets,
                            symptom_to_diseases, disease_to_genes, gene_to_variants):
    """Find a single variant in a gene associated with a phenotypically
    similar (but different) disease - a realistic differential-diagnosis
    hard negative. Returns None if no suitable candidate exists."""
    phenocopy_disease = find_phenocopy_disease(causal_disease_id, disease_symptom_sets, symptom_to_diseases)
    if phenocopy_disease is None:
        return None

    candidate_genes = [g for g in disease_to_genes.get(phenocopy_disease, []) if g != causal_gene]
    random.shuffle(candidate_genes)

    for gene in candidate_genes:
        gene_variants = gene_to_variants.get(gene)
        if gene_variants is not None and len(gene_variants) > 0:
            return gene_variants.sample(n=1, random_state=random.randint(0, 999999)).iloc[0]

    return None


def assign_phenotype_tier():
    """Assign each patient a realistic phenotyping-quality tier, instead of
    applying the same fixed noise level to every patient."""
    tiers = list(PHENOTYPE_TIERS.keys())
    weights = [PHENOTYPE_TIERS[t]["probability"] for t in tiers]
    return random.choices(tiers, weights=weights, k=1)[0]


def sample_patient_symptoms(disease_symptoms, hpo_df, keep_fraction_range, noise_range):
    """Sample a realistic, incomplete, noisy symptom set for a patient,
    using the keep-fraction and noise ranges for their assigned tier."""
    if len(disease_symptoms) == 0:
        return []

    disease_symptoms = disease_symptoms.drop_duplicates(subset=["hpo_id"]).reset_index(drop=True)

    keep_fraction = random.uniform(*keep_fraction_range)
    n_keep = max(1, int(len(disease_symptoms) * keep_fraction))
    n_sample = min(n_keep, len(disease_symptoms))

    weights = pd.to_numeric(disease_symptoms["frequency_numeric"], errors="coerce").fillna(0.3) + 0.01
    weights_list = weights.tolist()
    hpo_ids = disease_symptoms["hpo_id"].tolist()

    chosen_indices = set()
    attempts = 0
    while len(chosen_indices) < n_sample and attempts < n_sample * 20:
        pick = random.choices(range(len(hpo_ids)), weights=weights_list, k=1)[0]
        chosen_indices.add(pick)
        attempts += 1

    patient_symptoms = [hpo_ids[i] for i in chosen_indices]

    n_noise = random.randint(*noise_range)
    if n_noise > 0:
        random_symptoms = hpo_df[hpo_df["aspect"] == "P"].sample(n=n_noise, random_state=random.randint(0, 999999))
        patient_symptoms.extend(random_symptoms["hpo_id"].tolist())

    return list(set(patient_symptoms))


def assign_zygosity(inheritance_modes):
    """Assign a plausible zygosity based on the CAUSAL variant's inheritance mode."""
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


def assign_background_zygosity(bg_gene, gene_to_inheritance):
    """Assign zygosity that's typically INCONSISTENT with the background
    variant's own gene inheritance mode - a genuine inheritance red flag
    a real system should learn to weigh down, rather than unrelated noise."""
    true_mode = gene_to_inheritance.get(bg_gene, "")
    if pd.isna(true_mode):
        true_mode = ""
    true_mode = str(true_mode).lower()

    if "recessive" in true_mode:
        return "heterozygous"
    if "dominant" in true_mode:
        return random.choice(["homozygous", "heterozygous"])

    return random.choice(["heterozygous", "homozygous"])


def simulate_quality_metadata():
    return {
        "simulated_read_depth": random.randint(15, 100),
        "simulated_genotype_quality": random.randint(20, 99),
        "simulated_filter_status": random.choices(["PASS", "LOWQUAL"], weights=[0.95, 0.05])[0],
    }


def sample_background_variants(general_pool, hard_negative_pool, pathogenic_pool,
                                 causal_gene, phenocopy_row, n_background,
                                 n_uncertain_hn, n_pathogenic_hn):
    """Background variants: general random, plus THREE types of guaranteed
    hard negatives:
      1. Uncertain/conflicting significance, unrelated gene
      2. Genuinely pathogenic, but in an UNRELATED gene (incidental finding)
      3. A variant in a gene for a DIFFERENT disease that shares several
         symptoms with the true disease (phenocopy / differential diagnosis)."""

    hn_candidates = hard_negative_pool.sample(n=min(n_uncertain_hn + 2, len(hard_negative_pool)), random_state=random.randint(0, 999999))
    hn_uncertain = hn_candidates[hn_candidates["gene_symbol"] != causal_gene].head(n_uncertain_hn)

    path_candidates = pathogenic_pool.sample(n=min(n_pathogenic_hn + 2, len(pathogenic_pool)), random_state=random.randint(0, 999999))
    hn_pathogenic = path_candidates[path_candidates["gene_symbol"] != causal_gene].head(n_pathogenic_hn)

    if phenocopy_row is not None:
        phenocopy_df = pd.DataFrame([phenocopy_row])
    else:
        phenocopy_df = pd.DataFrame(columns=hn_uncertain.columns)

    total_hn = len(hn_uncertain) + len(hn_pathogenic) + len(phenocopy_df)
    remaining_needed = n_background - total_hn

    gen_candidates = general_pool.sample(n=min(remaining_needed + 5, len(general_pool)), random_state=random.randint(0, 999999))
    general_background = gen_candidates[gen_candidates["gene_symbol"] != causal_gene].head(remaining_needed)

    hard_negative_ids = set(hn_uncertain["variant_id"]) | set(hn_pathogenic["variant_id"])
    if len(phenocopy_df):
        hard_negative_ids |= set(phenocopy_df["variant_id"])

    background_df = pd.concat([hn_uncertain, hn_pathogenic, phenocopy_df, general_background], ignore_index=True)
    background_df = background_df.drop_duplicates(subset=["variant_id"])

    return background_df, hard_negative_ids


def generate_patients(snv, hpo, clinvar, omim_df, n_patients):
    eligible = build_eligible_causal_pool(snv)
    print(f"Eligible causal variant pool: {len(eligible)} rows, {eligible['gene_symbol'].nunique()} unique genes")

    print("Precomputing background variant pools...")
    general_pool = clinvar[clinvar["clinvar_significance"].isin(["Benign", "Likely_benign", "Uncertain_significance"])]
    hard_negative_pool = clinvar[clinvar["clinvar_significance"].isin(["Uncertain_significance", "Conflicting"])]
    pathogenic_pool = clinvar[clinvar["clinvar_significance"].isin(["Pathogenic", "Likely_pathogenic"])]

    print("Precomputing gene -> variants lookup (for phenocopy selection)...")
    gene_to_variants = build_gene_to_variants_lookup(clinvar)

    print("Precomputing gene -> inheritance mode lookup...")
    gene_to_inheritance = build_gene_to_inheritance_lookup(omim_df)

    print("Precomputing HPO symptom lookup...")
    hpo_lookup = build_hpo_lookup(hpo)
    all_p_symptoms = hpo[hpo["aspect"] == "P"]

    print("Precomputing disease symptom sets and phenocopy index...")
    disease_symptom_sets = build_disease_symptom_sets(hpo)
    symptom_to_diseases = build_symptom_to_diseases_index(disease_symptom_sets)
    disease_to_genes = build_disease_to_gene_lookup(omim_df)

    causal_sample = sample_causal_variants(eligible, int(n_patients * 1.2), MAX_PATIENTS_PER_GENE)
    print(f"Sampled {len(causal_sample)} candidate causal variants across {causal_sample['gene_symbol'].nunique()} unique genes")

    patients_meta = []
    patient_variants = []
    skipped_no_symptoms = 0
    no_phenocopy_found = 0

    patient_id = 0
    for _, causal_row in causal_sample.iterrows():
        if patient_id >= n_patients:
            break

        disease_id = f"OMIM:{int(causal_row['phenotype_mim_number'])}"
        disease_symptoms = hpo_lookup.get(disease_id, pd.DataFrame())

        phenotype_tier = assign_phenotype_tier()
        tier_config = PHENOTYPE_TIERS[phenotype_tier]

        patient_symptoms = sample_patient_symptoms(
            disease_symptoms, all_p_symptoms,
            tier_config["keep_fraction"], tier_config["noise_symptoms"]
        )

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
            "phenotype_tier": phenotype_tier,
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

        phenocopy_row = find_phenocopy_variant(
            disease_id, causal_row["gene_symbol"], disease_symptom_sets,
            symptom_to_diseases, disease_to_genes, gene_to_variants
        )
        if phenocopy_row is None:
            no_phenocopy_found += 1

        background_df, hard_negative_ids = sample_background_variants(
            general_pool, hard_negative_pool, pathogenic_pool,
            causal_row["gene_symbol"], phenocopy_row, N_BACKGROUND_VARIANTS,
            N_HARD_NEGATIVES_UNCERTAIN, N_HARD_NEGATIVES_PATHOGENIC
        )

        for _, bg_row in background_df.iterrows():
            bg_quality = simulate_quality_metadata()
            patient_variants.append({
                "patient_id": patient_id,
                "variant_id": bg_row["variant_id"],
                "gene_symbol": bg_row["gene_symbol"],
                "is_causal": 0,
                "is_hard_negative": 1 if bg_row["variant_id"] in hard_negative_ids else 0,
                "zygosity": assign_background_zygosity(bg_row["gene_symbol"], gene_to_inheritance),
                **bg_quality
            })

        if patient_id % 500 == 0:
            print(f"  ...generated {patient_id} patients")

    print(f"Skipped candidates with no HPO symptom coverage: {skipped_no_symptoms}")
    print(f"Patients with no phenocopy hard negative found: {no_phenocopy_found}")
    if patient_id < n_patients:
        print(f"WARNING: only generated {patient_id}/{n_patients} patients")

    return pd.DataFrame(patients_meta), pd.DataFrame(patient_variants)


if __name__ == "__main__":
    snv, hpo, clinvar, omim_df = load_data()

    patients_meta_df, patient_variants_df = generate_patients(snv, hpo, clinvar, omim_df, N_PATIENTS)

    print()
    print(f"Total patients generated: {len(patients_meta_df)}")
    print(f"Total patient-variant rows: {len(patient_variants_df)}")
    print(f"Average variants per patient: {len(patient_variants_df) / len(patients_meta_df):.1f}")
    print(f"Patients with 0 symptoms sampled: {(patients_meta_df['n_patient_symptoms'] == 0).sum()}")
    print(f"Hard negatives total: {patient_variants_df['is_hard_negative'].sum()}")
    print()
    print("Phenotype tier distribution:")
    print(patients_meta_df["phenotype_tier"].value_counts())

    patients_meta_df.to_csv("data/processed/snv/patients_meta.csv", index=False)
    patient_variants_df.to_csv("data/processed/snv/patient_variants.csv", index=False)