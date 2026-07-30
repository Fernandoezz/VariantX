import random
import pandas as pd


random.seed(42)

N_CNV_PATIENTS = 1000        # real training target (proposal range: 500-2000)
MAX_PATIENTS_PER_GENE = 3
N_BACKGROUND_DELETIONS = 30  # smaller than SNV's 100 - CNV candidate lists are naturally shorter
N_HARD_NEGATIVES = 3
SYMPTOM_KEEP_FRACTION = (0.6, 0.9)
N_NOISE_SYMPTOMS = (0, 2)


def load_data():
    cnv = pd.read_csv("data/processed/cnv/cnv_features.csv", dtype={"chromosome": str}, low_memory=False)
    population = pd.read_csv("data/interim/gnomad_sv_tables/population_frequency_clean.csv", dtype={"chromosome": str})
    hpo = pd.read_csv("data/interim/hpo_tables/disease_hpo_annotations_clean.csv", low_memory=False)
    return cnv, population, hpo


def build_eligible_causal_pool(cnv):
    """For each deletion, keep only its highest-haploinsufficiency gene match
    as the primary disease-driving gene - one row per deletion, not per
    overlapping gene."""
    eligible = cnv[
        cnv["disease_name"].notna() &
        cnv["phenotype_type"].isin(["confirmed_mendelian", "unconfirmed_mendelian"]) &
        cnv["haploinsufficiency_score"].notna()
    ].copy()

    eligible = eligible.sort_values("haploinsufficiency_score", ascending=False)
    eligible = eligible.drop_duplicates(subset=["variant_id"], keep="first")

    return eligible


def sample_causal_deletions(eligible, n_patients, max_per_gene):
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
    filtered = hpo_df[(hpo_df["aspect"] == "P") & (hpo_df["is_negated"] == False)]
    return {disease_id: group for disease_id, group in filtered.groupby("disease_id")}


def sample_patient_symptoms(disease_symptoms, hpo_df):
    if len(disease_symptoms) == 0:
        return []

    disease_symptoms = disease_symptoms.drop_duplicates(subset=["hpo_id"]).reset_index(drop=True)

    keep_fraction = random.uniform(*SYMPTOM_KEEP_FRACTION)
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

    n_noise = random.randint(*N_NOISE_SYMPTOMS)
    if n_noise > 0:
        random_symptoms = hpo_df.sample(n=n_noise, random_state=random.randint(0, 999999))
        patient_symptoms.extend(random_symptoms["hpo_id"].tolist())

    return list(set(patient_symptoms))


def assign_zygosity(inheritance_modes):
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


def sample_background_deletions(genic_pool, intergenic_pool, hard_negative_pool, causal_gene, n_background, n_hard_negatives):
    """Sample background from a realistic mix: some genic (with real gene
    features), some intergenic (genuinely no gene overlap) - plus guaranteed
    hard negatives. Sample first, exclude causal gene after."""

    hard_negatives = hard_negative_pool.sample(n=min(n_hard_negatives + 2, len(hard_negative_pool)), random_state=random.randint(0, 999999))
    hard_negatives = hard_negatives[hard_negatives["gene_symbol"] != causal_gene].head(n_hard_negatives)

    remaining_needed = n_background - len(hard_negatives)

    # Roughly match real biology: ~57% genic, ~43% intergenic among the rest
    n_genic = int(remaining_needed * 0.57)
    n_intergenic = remaining_needed - n_genic

    genic_background = genic_pool.sample(n=min(n_genic + 5, len(genic_pool)), random_state=random.randint(0, 999999))
    genic_background = genic_background[genic_background["gene_symbol"] != causal_gene].head(n_genic)

    intergenic_background = intergenic_pool.sample(n=min(n_intergenic + 5, len(intergenic_pool)), random_state=random.randint(0, 999999))
    intergenic_background = intergenic_background.head(n_intergenic)

    hard_negative_ids = set(hard_negatives["variant_id"])

    hn_slim = hard_negatives[["variant_id", "gene_symbol"]].copy()
    genic_slim = genic_background[["variant_id", "gene_symbol"]].copy()
    inter_slim = intergenic_background[["variant_id"]].copy()
    inter_slim["gene_symbol"] = None

    background_df = pd.concat([hn_slim, genic_slim, inter_slim], ignore_index=True)
    background_df = background_df.drop_duplicates(subset=["variant_id"])

    return background_df, hard_negative_ids


def generate_cnv_patients(cnv, population, hpo, n_patients):
    eligible = build_eligible_causal_pool(cnv)
    print(f"Eligible causal deletion pool: {len(eligible)} deletions, {eligible['gene_symbol'].nunique()} unique genes")

    print("Precomputing background/hard-negative pools...")
    hard_negative_pool = cnv[cnv["haploinsufficiency_score"].notna()]

    # Genic background: any deletion known to overlap a gene (from cnv_features.csv)
    genic_pool = cnv.drop_duplicates(subset=["variant_id"])

    # Intergenic background: population deletions that never appear in cnv_features.csv at all
    genic_variant_ids = set(cnv["variant_id"])
    intergenic_pool = population[~population["variant_id"].isin(genic_variant_ids)]

    print(f"Genic background pool: {len(genic_pool)}, Intergenic background pool: {len(intergenic_pool)}")

    print("Precomputing HPO symptom lookup...")
    hpo_lookup = build_hpo_lookup(hpo)
    all_p_symptoms = hpo[hpo["aspect"] == "P"]

    causal_sample = sample_causal_deletions(eligible, int(n_patients * 1.2), MAX_PATIENTS_PER_GENE)
    print(f"Sampled {len(causal_sample)} candidate causal deletions across {causal_sample['gene_symbol'].nunique()} unique genes")

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

        background_df, hard_negative_ids = sample_background_deletions(
            genic_pool, intergenic_pool, hard_negative_pool,
            causal_row["gene_symbol"], N_BACKGROUND_DELETIONS, N_HARD_NEGATIVES
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

        if patient_id % 100 == 0:
            print(f"  ...generated {patient_id} patients")

    print(f"Skipped candidates with no HPO symptom coverage: {skipped_no_symptoms}")
    if patient_id < n_patients:
        print(f"WARNING: only generated {patient_id}/{n_patients} patients")

    return pd.DataFrame(patients_meta), pd.DataFrame(patient_variants)


if __name__ == "__main__":
    cnv, population, hpo = load_data()

    patients_meta_df, patient_variants_df = generate_cnv_patients(cnv, population, hpo, N_CNV_PATIENTS)

    print()
    print(f"Total patients generated: {len(patients_meta_df)}")
    print(f"Total patient-variant rows: {len(patient_variants_df)}")
    print(f"Average variants per patient: {len(patient_variants_df) / len(patients_meta_df):.1f}")
    print(f"Patients with 0 symptoms sampled: {(patients_meta_df['n_patient_symptoms'] == 0).sum()}")
    print(f"Hard negatives total: {patient_variants_df['is_hard_negative'].sum()}")

    patients_meta_df.to_csv("data/processed/cnv/patients_meta.csv", index=False)
    patient_variants_df.to_csv("data/processed/cnv/patient_variants.csv", index=False)