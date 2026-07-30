import gzip
import pandas as pd
import numpy as np
import lightgbm as lgb


# ============================================================
# Shared VCF parsing / reference loading
# ============================================================

def parse_patient_vcf(filepath):
    """
    Parse a patient VCF (plain text or gzipped) into a DataFrame of SNV/indel
    variants. Handles both raw .vcf and .vcf.gz.
    """
    open_func = gzip.open if filepath.endswith(".gz") else open
    mode = "rt" if filepath.endswith(".gz") else "r"

    rows = []
    with open_func(filepath, mode) as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, variant_id, ref, alt, qual, filter_status, info = fields[:8]

            genotype = None
            if len(fields) >= 10:
                format_fields = fields[8].split(":")
                sample_fields = fields[9].split(":")
                if "GT" in format_fields:
                    gt_index = format_fields.index("GT")
                    genotype = sample_fields[gt_index]

            rows.append({
                "chromosome": chrom,
                "position": int(pos),
                "reference_allele": ref,
                "alternate_allele": alt,
                "filter_status": filter_status,
                "genotype_raw": genotype,
                "variant_id": f"{chrom}_{pos}_{ref}_{alt}",
            })

    return pd.DataFrame(rows)


def parse_patient_cnv_vcf(filepath):
    """
    Parse a CNV-style VCF - expects SVTYPE=DEL and END in the INFO field,
    similar to gnomAD-SV's format. Skips any record that isn't a deletion
    or is missing the END coordinate.
    """
    open_func = gzip.open if filepath.endswith(".gz") else open
    mode = "rt" if filepath.endswith(".gz") else "r"

    rows = []
    with open_func(filepath, mode) as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, variant_id, ref, alt, qual, filter_status, info = fields[:8]

            info_dict = {}
            for item in info.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    info_dict[k] = v

            sv_type = info_dict.get("SVTYPE", "")
            if sv_type != "DEL":
                continue

            end_pos = info_dict.get("END")
            if end_pos is None:
                continue

            genotype = None
            if len(fields) >= 10:
                format_fields = fields[8].split(":")
                sample_fields = fields[9].split(":")
                if "GT" in format_fields:
                    genotype = sample_fields[format_fields.index("GT")]

            clean_chrom = chrom.replace("chr", "")
            start = int(pos)
            end = int(end_pos)

            rows.append({
                "chromosome": clean_chrom,
                "start_position": start,
                "end_position": end,
                "deletion_size": abs(end - start),
                "allele_frequency": None,
                "filter_status": filter_status,
                "genotype_raw": genotype,
                "variant_id": f"{clean_chrom}_{start}_{end}",
            })

    return pd.DataFrame(rows)


def load_reference_tables():
    """Load all the reference tables needed for live annotation.
    Called once when the app starts, not per-request."""
    clinvar = pd.read_csv(
        "data/interim/clinvar_tables/clinvar_variants_clean.csv",
        dtype={"chromosome": str}
    )
    ensembl = pd.read_csv(
        "data/interim/ensembl_tables/gene_coordinates_clean.csv",
        dtype={"chromosome": str}
    )
    gnomad = pd.read_csv("data/interim/gnomad_tables/gene_constraint_clean.csv")
    clingen = pd.read_csv("data/interim/clingen_tables/gene_dosage_clean.csv")
    omim = pd.read_csv("data/interim/omim_tables/gene_disease_inheritance.csv")

    return {
        "clinvar": clinvar,
        "ensembl": ensembl,
        "gnomad": gnomad,
        "clingen": clingen,
        "omim": omim,
    }


# ============================================================
# SNV annotation
# ============================================================

def resolve_gene_by_coordinate(chromosome, position, ensembl_df):
    """Fallback gene lookup for variants not found in ClinVar - checks
    which gene's coordinate range contains this position."""
    matches = ensembl_df[
        (ensembl_df["chromosome"] == chromosome) &
        (ensembl_df["start_position"] <= position) &
        (ensembl_df["end_position"] >= position)
    ]
    if len(matches) == 0:
        return None
    return matches.iloc[0]["gene_symbol"]


def annotate_variants(patient_variants_df, reference_tables):
    """Attach gene, clinical significance, and gene-level evidence to each
    parsed SNV/indel patient variant."""
    clinvar = reference_tables["clinvar"]
    ensembl = reference_tables["ensembl"]
    gnomad = reference_tables["gnomad"]
    clingen = reference_tables["clingen"]

    annotated_rows = []

    for _, variant in patient_variants_df.iterrows():
        clinvar_match = clinvar[clinvar["variant_id"] == variant["variant_id"]]

        if len(clinvar_match) > 0:
            match = clinvar_match.iloc[0]
            gene_symbol = match["gene_symbol"]
            clinvar_significance = match["clinvar_significance"]
            review_status = match["review_status"]
            has_multiple_genes = match["has_multiple_genes"]
            in_clinvar = True
        else:
            gene_symbol = resolve_gene_by_coordinate(
                variant["chromosome"], variant["position"], ensembl
            )
            clinvar_significance = None
            review_status = None
            has_multiple_genes = False
            in_clinvar = False

        row = variant.to_dict()
        row["gene_symbol"] = gene_symbol
        row["clinvar_significance"] = clinvar_significance
        row["review_status"] = review_status
        row["has_multiple_genes"] = has_multiple_genes
        row["in_clinvar"] = in_clinvar

        annotated_rows.append(row)

    annotated_df = pd.DataFrame(annotated_rows)

    ensembl_dedup = ensembl.drop_duplicates(subset=["gene_symbol"], keep="first")
    ensembl_dedup = ensembl_dedup[["gene_symbol", "start_position", "end_position", "gene_biotype"]]
    ensembl_dedup = ensembl_dedup.rename(columns={
        "start_position": "gene_start_position",
        "end_position": "gene_end_position"
    })

    annotated_df = annotated_df.merge(ensembl_dedup, on="gene_symbol", how="left")
    annotated_df = annotated_df.merge(gnomad, on="gene_symbol", how="left")
    annotated_df = annotated_df.merge(clingen, on="gene_symbol", how="left")

    return annotated_df


# ============================================================
# CNV annotation
# ============================================================

def annotate_cnv_variants(patient_deletions_df, reference_tables):
    """
    For each patient CNV deletion, find overlapping genes via coordinate
    range (same logic as the batch build_cnv_features.py overlap join),
    then attach gene-level evidence. A deletion overlapping multiple genes
    produces multiple rows, same as the batch pipeline.
    """
    ensembl = reference_tables["ensembl"]
    gnomad = reference_tables["gnomad"]
    clingen = reference_tables["clingen"]

    records = []

    for _, deletion in patient_deletions_df.iterrows():
        gene_matches = ensembl[
            (ensembl["chromosome"] == deletion["chromosome"]) &
            (ensembl["start_position"] <= deletion["end_position"]) &
            (ensembl["end_position"] >= deletion["start_position"])
        ]

        if len(gene_matches) == 0:
            records.append({
                "variant_id": deletion["variant_id"],
                "chromosome": deletion["chromosome"],
                "deletion_start": deletion["start_position"],
                "deletion_end": deletion["end_position"],
                "deletion_size": deletion["deletion_size"],
                "allele_frequency": deletion.get("allele_frequency", None),
                "gene_symbol": None,
                "gene_start_position": None,
                "gene_end_position": None,
                "gene_biotype": None,
                "overlap_length": 0,
                "overlap_fraction_of_gene": None,
                "overlap_fraction_of_deletion": None,
                "genotype_raw": deletion.get("genotype_raw"),
                "filter_status": deletion.get("filter_status"),
            })
            continue

        for _, gene in gene_matches.iterrows():
            overlap_start = max(deletion["start_position"], gene["start_position"])
            overlap_end = min(deletion["end_position"], gene["end_position"])
            overlap_length = overlap_end - overlap_start

            gene_length = gene["end_position"] - gene["start_position"]
            deletion_length = deletion["end_position"] - deletion["start_position"]

            records.append({
                "variant_id": deletion["variant_id"],
                "chromosome": deletion["chromosome"],
                "deletion_start": deletion["start_position"],
                "deletion_end": deletion["end_position"],
                "deletion_size": deletion["deletion_size"],
                "allele_frequency": deletion.get("allele_frequency", None),
                "gene_symbol": gene["gene_symbol"],
                "gene_start_position": gene["start_position"],
                "gene_end_position": gene["end_position"],
                "gene_biotype": gene["gene_biotype"],
                "overlap_length": overlap_length,
                "overlap_fraction_of_gene": overlap_length / gene_length if gene_length > 0 else None,
                "overlap_fraction_of_deletion": overlap_length / deletion_length if deletion_length > 0 else None,
                "genotype_raw": deletion.get("genotype_raw"),
                "filter_status": deletion.get("filter_status"),
            })

    annotated_df = pd.DataFrame(records)

    annotated_df = annotated_df.merge(gnomad, on="gene_symbol", how="left")
    annotated_df = annotated_df.merge(clingen, on="gene_symbol", how="left")

    return annotated_df


# ============================================================
# Phenotype scoring (shared between SNV and CNV)
# ============================================================

def compute_phenotype_score(patient_hpo_terms, gene_symbol, omim_df, hpo_annotations_df):
    """
    Score how well the patient's entered symptoms match the best candidate
    disease associated with this gene. Same formula as the batch pipeline:
    matched symptom weight / total disease symptom weight, best disease wins.
    """
    if gene_symbol is None or pd.isna(gene_symbol):
        return 0.0, None

    gene_diseases = omim_df[omim_df["gene_symbol"] == gene_symbol]
    if len(gene_diseases) == 0 or not patient_hpo_terms:
        return 0.0, None

    patient_symptom_set = set(patient_hpo_terms)
    best_score = 0.0
    best_disease = None

    for _, disease_row in gene_diseases.dropna(subset=["phenotype_mim_number"]).iterrows():
        disease_id = f"OMIM:{int(disease_row['phenotype_mim_number'])}"

        disease_symptoms = hpo_annotations_df[
            (hpo_annotations_df["disease_id"] == disease_id) &
            (hpo_annotations_df["aspect"] == "P") &
            (hpo_annotations_df["is_negated"] == False)
        ]

        if len(disease_symptoms) == 0:
            continue

        freq = pd.to_numeric(disease_symptoms["frequency_numeric"], errors="coerce").fillna(0.3)
        total_weight = freq.sum()
        if total_weight == 0:
            continue

        matched_weight = disease_symptoms.loc[
            disease_symptoms["hpo_id"].isin(patient_symptom_set), "frequency_numeric"
        ].fillna(0.3).sum()

        score = matched_weight / total_weight

        if score > best_score:
            best_score = score
            best_disease = disease_row["disease_name"]

    return best_score, best_disease


def score_patient_variants(annotated_df, patient_hpo_terms, omim_df, hpo_annotations_df):
    """Add phenotype_similarity_score and best_matching_disease to every
    annotated variant, based on the physician's entered symptoms."""
    scores = []
    diseases = []

    for _, row in annotated_df.iterrows():
        score, disease = compute_phenotype_score(
            patient_hpo_terms, row["gene_symbol"], omim_df, hpo_annotations_df
        )
        scores.append(score)
        diseases.append(disease)

    annotated_df = annotated_df.copy()
    annotated_df["phenotype_similarity_score"] = scores
    annotated_df["best_matching_disease"] = diseases

    return annotated_df


# ============================================================
# Shared prediction helpers
# ============================================================

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


def assign_zygosity_from_genotype(genotype_raw):
    """Convert VCF genotype notation (e.g. '1/1', '0/1') into the
    heterozygous/homozygous categories the models were trained on."""
    if pd.isna(genotype_raw):
        return "heterozygous"

    alleles = genotype_raw.replace("|", "/").split("/")
    if len(alleles) == 2 and alleles[0] == alleles[1] and alleles[0] != "0":
        return "homozygous"
    return "heterozygous"


# ============================================================
# SNV prediction
# ============================================================

SNV_MODEL_FEATURE_COLUMNS = [
    "zygosity", "simulated_read_depth", "simulated_genotype_quality", "simulated_filter_status",
    "phenotype_similarity_score", "has_multiple_genes", "clinvar_significance", "review_status",
    "oe_lof", "oe_lof_lower", "oe_lof_upper", "oe_lof_upper_rank", "pLI",
    "haploinsufficiency_score", "triplosensitivity_score",
    "gene_start_position", "gene_end_position", "gene_biotype",
    "SIFT_score", "SIFT_converted_rankscore", "SIFT_pred",
    "Polyphen2_HDIV_score", "Polyphen2_HDIV_rankscore", "Polyphen2_HDIV_pred",
    "Polyphen2_HVAR_score", "Polyphen2_HVAR_rankscore", "Polyphen2_HVAR_pred",
    "CADD_raw", "CADD_raw_rankscore", "CADD_phred",
    "GERP_NR", "GERP_RS",
    "phyloP100way_vertebrate", "phyloP100way_vertebrate_rankscore",
    "phastCons100way_vertebrate", "phastCons100way_vertebrate_rankscore"
]

SNV_CATEGORICAL_COLUMNS = [
    "clinvar_significance", "review_status", "zygosity", "gene_biotype",
    "SIFT_pred", "Polyphen2_HDIV_pred", "Polyphen2_HVAR_pred", "simulated_filter_status"
]


def prepare_for_prediction(scored_df):
    """Fill in fields the SNV model expects that a live VCF won't naturally
    provide (dbNSFP scores, simulated quality metrics), using neutral/
    missing defaults, then align columns to match training exactly."""
    df = scored_df.copy()

    df["zygosity"] = df["genotype_raw"].apply(assign_zygosity_from_genotype)

    df["simulated_read_depth"] = 50
    df["simulated_genotype_quality"] = 60
    df["simulated_filter_status"] = df["filter_status"].fillna("PASS")

    for col in [
        "SIFT_score", "SIFT_converted_rankscore", "SIFT_pred",
        "Polyphen2_HDIV_score", "Polyphen2_HDIV_rankscore", "Polyphen2_HDIV_pred",
        "Polyphen2_HVAR_score", "Polyphen2_HVAR_rankscore", "Polyphen2_HVAR_pred",
        "CADD_raw", "CADD_raw_rankscore", "CADD_phred",
        "GERP_NR", "GERP_RS",
        "phyloP100way_vertebrate", "phyloP100way_vertebrate_rankscore",
        "phastCons100way_vertebrate", "phastCons100way_vertebrate_rankscore"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["SIFT_score", "Polyphen2_HDIV_score", "Polyphen2_HVAR_score"]:
        df[col] = df[col].apply(parse_multi_value_score)

    for col in SNV_CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    X = df[SNV_MODEL_FEATURE_COLUMNS]
    return X, df


def predict_and_rank(scored_df, model_path="data/processed/snv/snv_model.txt"):
    model = lgb.Booster(model_file=model_path)

    X, df = prepare_for_prediction(scored_df)
    predictions = model.predict(X)

    df = df.copy()
    df["relevance_score"] = predictions
    df = df.sort_values("relevance_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


# ============================================================
# CNV prediction
# ============================================================

CNV_MODEL_FEATURE_COLUMNS = [
    "zygosity", "simulated_read_depth", "simulated_genotype_quality", "simulated_filter_status",
    "phenotype_similarity_score", "deletion_size", "allele_frequency",
    "gene_start_position", "gene_end_position", "gene_biotype",
    "overlap_length", "overlap_fraction_of_gene", "overlap_fraction_of_deletion",
    "oe_lof", "oe_lof_lower", "oe_lof_upper", "oe_lof_upper_rank", "pLI",
    "haploinsufficiency_score", "triplosensitivity_score"
]

CNV_CATEGORICAL_COLUMNS = ["zygosity", "gene_biotype", "simulated_filter_status"]


def prepare_cnv_for_prediction(scored_df):
    df = scored_df.copy()

    df["zygosity"] = df["genotype_raw"].apply(assign_zygosity_from_genotype)
    df["simulated_read_depth"] = 50
    df["simulated_genotype_quality"] = 60
    df["simulated_filter_status"] = df["filter_status"].fillna("PASS")

    if "allele_frequency" not in df.columns or df["allele_frequency"].isna().all():
        df["allele_frequency"] = 0.0001  # neutral default: treat as rare if unknown
    else:
        df["allele_frequency"] = df["allele_frequency"].fillna(0.0001)

    for col in CNV_CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    X = df[CNV_MODEL_FEATURE_COLUMNS]
    return X, df


def predict_and_rank_cnv(scored_df, model_path="data/processed/cnv/cnv_model.txt"):
    model = lgb.Booster(model_file=model_path)

    X, df = prepare_cnv_for_prediction(scored_df)
    predictions = model.predict(X)

    df = df.copy()
    df["relevance_score"] = predictions
    df = df.sort_values("relevance_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


if __name__ == "__main__":
    patient_variants = parse_patient_vcf("data/sample_patient/sample_patient.vcf")
    reference_tables = load_reference_tables()
    hpo_annotations = pd.read_csv("data/interim/hpo_tables/disease_hpo_annotations_clean.csv", low_memory=False)

    annotated = annotate_variants(patient_variants, reference_tables)

    example_patient_symptoms = ["HP:0003002"]  # Breast carcinoma
    scored = score_patient_variants(annotated, example_patient_symptoms, reference_tables["omim"], hpo_annotations)

    ranked = predict_and_rank(scored)

    print("\n=== Ranked Results (SNV) ===")
    print(ranked[["rank", "gene_symbol", "relevance_score", "phenotype_similarity_score", "best_matching_disease"]])