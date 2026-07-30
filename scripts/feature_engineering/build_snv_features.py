import pandas as pd


def load_tables():
    clinvar = pd.read_csv(
        "data/interim/clinvar_tables/clinvar_variants_clean.csv",
        dtype={"chromosome": str}
    )
    gnomad = pd.read_csv("data/interim/gnomad_tables/gene_constraint_clean.csv")
    clingen = pd.read_csv("data/interim/clingen_tables/gene_dosage_clean.csv")
    ensembl = pd.read_csv("data/interim/ensembl_tables/gene_coordinates_clean.csv")
    omim = pd.read_csv("data/interim/omim_tables/gene_disease_inheritance.csv")

    return clinvar, gnomad, clingen, ensembl, omim


def build_snv_feature_matrix():
    clinvar, gnomad, clingen, ensembl, omim = load_tables()

    print(f"Step 1 - ClinVar backbone: {len(clinvar)} rows")

    # Step 2: gnomAD constraint (one row per gene - safe one-to-one join)
    df = clinvar.merge(gnomad, on="gene_symbol", how="left")
    print(f"Step 2 - after gnomAD constraint join: {len(df)} rows")

    # Step 3: ClinGen dosage (one row per gene - safe one-to-one join)
    df = df.merge(clingen, on="gene_symbol", how="left")
    print(f"Step 3 - after ClinGen dosage join: {len(df)} rows")

    # Step 4: Ensembl coordinates - deduplicated by gene_symbol for this join only.
    # (Multi-locus genes like 5S_rRNA are not disease genes relevant to ClinVar,
    # so collapsing to one location per symbol here is safe and prevents
    # accidental row multiplication.)
    ensembl_dedup = ensembl.drop_duplicates(subset=["gene_symbol"], keep="first")
    ensembl_dedup = ensembl_dedup[["gene_symbol", "start_position", "end_position", "gene_biotype"]]
    ensembl_dedup = ensembl_dedup.rename(columns={
        "start_position": "gene_start_position",
        "end_position": "gene_end_position"
    })

    df = df.merge(ensembl_dedup, on="gene_symbol", how="left")
    print(f"Step 4 - after Ensembl coordinates join: {len(df)} rows")

    # Step 5: OMIM gene-disease-inheritance - deliberately NOT deduplicated.
    # A gene can cause multiple diseases; each is a distinct clinical hypothesis
    # with its own inheritance mode, so this join intentionally expands rows -
    # one row per (variant, gene, candidate disease) combination.
    df = df.merge(omim, on="gene_symbol", how="left")
    print(f"Step 5 - after OMIM gene-disease-inheritance join: {len(df)} rows")

    return df


if __name__ == "__main__":
    feature_df = build_snv_feature_matrix()

    print()
    print(f"Final SNV feature matrix: {len(feature_df)} rows, {len(feature_df.columns)} columns")
    print(f"Columns: {feature_df.columns.tolist()}")
    print()
    print(f"Unique variant_id count: {feature_df['variant_id'].nunique()}")
    print(f"Rows with no OMIM disease match: {feature_df['disease_name'].isna().sum()}")

    feature_df.to_csv("data/processed/snv/snv_features.csv", index=False)