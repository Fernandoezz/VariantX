import pandas as pd


SNV_GENE_VARIANT_COLUMNS = [
    "variant_id", "chromosome", "position", "reference_allele", "alternate_allele",
    "gene_symbol", "has_multiple_genes", "clinvar_significance", "review_status",
    "oe_lof", "oe_lof_lower", "oe_lof_upper", "oe_lof_upper_rank", "pLI",
    "haploinsufficiency_score", "triplosensitivity_score",
    "gene_start_position", "gene_end_position", "gene_biotype"
]

CNV_GENE_VARIANT_COLUMNS = [
    "variant_id", "chromosome", "deletion_start", "deletion_end", "deletion_size",
    "allele_frequency", "gene_symbol", "gene_start_position", "gene_end_position",
    "gene_biotype", "overlap_length", "overlap_fraction_of_gene", "overlap_fraction_of_deletion",
    "oe_lof", "oe_lof_lower", "oe_lof_upper", "oe_lof_upper_rank", "pLI",
    "haploinsufficiency_score", "triplosensitivity_score"
]


def dedupe_feature_table(feature_df, key_columns, keep_columns):
    """Collapse a variant-gene-disease table down to one row per
    variant_id + gene_symbol, dropping disease-specific columns since
    patient-level phenotype scoring already resolved the relevant disease."""
    deduped = feature_df.drop_duplicates(subset=key_columns, keep="first")
    return deduped[keep_columns]


def assemble_snv_training_table():
    print("=== Assembling SNV training table ===")

    patient_variants = pd.read_csv("data/processed/snv/patient_variants_scored.csv")
    snv_features = pd.read_csv("data/processed/snv/snv_features.csv", dtype={"chromosome": str}, low_memory=False)

    print(f"Patient-variant rows: {len(patient_variants)}")
    print(f"Raw SNV feature rows (pre-dedupe): {len(snv_features)}")

    gene_variant_features = dedupe_feature_table(
        snv_features, ["variant_id", "gene_symbol"], SNV_GENE_VARIANT_COLUMNS
    )
    print(f"Deduped gene-variant feature rows: {len(gene_variant_features)}")

    training_df = patient_variants.merge(
        gene_variant_features, on=["variant_id", "gene_symbol"], how="left"
    )

    print(f"Final SNV training table: {len(training_df)} rows")
    print(f"Rows with no matching feature data: {training_df['clinvar_significance'].isna().sum()}")

    training_df.to_csv("data/processed/snv/snv_training_table.csv", index=False)
    return training_df


def assemble_cnv_training_table():
    print("\n=== Assembling CNV training table ===")

    patient_variants = pd.read_csv("data/processed/cnv/patient_variants_scored.csv")
    cnv_features = pd.read_csv("data/processed/cnv/cnv_features.csv", dtype={"chromosome": str}, low_memory=False)

    print(f"Patient-variant rows: {len(patient_variants)}")
    print(f"Raw CNV feature rows (pre-dedupe): {len(cnv_features)}")

    gene_variant_features = dedupe_feature_table(
        cnv_features, ["variant_id", "gene_symbol"], CNV_GENE_VARIANT_COLUMNS
    )
    print(f"Deduped gene-variant feature rows: {len(gene_variant_features)}")

    training_df = patient_variants.merge(
        gene_variant_features, on=["variant_id", "gene_symbol"], how="left"
    )

    print(f"Final CNV training table: {len(training_df)} rows")
    print(f"Rows with no matching feature data (likely intergenic background): {training_df['overlap_length'].isna().sum()}")

    training_df.to_csv("data/processed/cnv/cnv_training_table.csv", index=False)
    return training_df


if __name__ == "__main__":
    snv_training = assemble_snv_training_table()
    cnv_training = assemble_cnv_training_table()

    print()
    print(f"SNV training table: {len(snv_training)} rows, {len(snv_training.columns)} columns")
    print(f"CNV training table: {len(cnv_training)} rows, {len(cnv_training.columns)} columns")