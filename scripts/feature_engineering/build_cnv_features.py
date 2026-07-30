import pandas as pd


def load_tables():
    cnv = pd.read_csv("data/interim/gnomad_sv_tables/population_frequency_clean.csv", dtype={"chromosome": str})
    ensembl = pd.read_csv("data/interim/ensembl_tables/gene_coordinates_clean.csv", dtype={"chromosome": str})
    gnomad = pd.read_csv("data/interim/gnomad_tables/gene_constraint_clean.csv")
    clingen = pd.read_csv("data/interim/clingen_tables/gene_dosage_clean.csv")
    omim = pd.read_csv("data/interim/omim_tables/gene_disease_inheritance.csv")

    return cnv, ensembl, gnomad, clingen, omim


def overlap_join_chromosome(cnv_chr_df, gene_chr_df):
    records = []

    for _, deletion in cnv_chr_df.iterrows():
        overlapping = gene_chr_df[
            (gene_chr_df["start_position"] <= deletion["end_position"]) &
            (gene_chr_df["end_position"] >= deletion["start_position"])
        ]

        for _, gene in overlapping.iterrows():
            overlap_start = max(deletion["start_position"], gene["start_position"])
            overlap_end = min(deletion["end_position"], gene["end_position"])
            overlap_length = overlap_end - overlap_start

            gene_length = gene["end_position"] - gene["start_position"]
            deletion_length = deletion["end_position"] - deletion["start_position"]

            overlap_fraction_of_gene = overlap_length / gene_length if gene_length > 0 else None
            overlap_fraction_of_deletion = overlap_length / deletion_length if deletion_length > 0 else None

            records.append({
                "variant_id": deletion["variant_id"],
                "chromosome": deletion["chromosome"],
                "deletion_start": deletion["start_position"],
                "deletion_end": deletion["end_position"],
                "deletion_size": deletion["deletion_size"],
                "allele_frequency": deletion["allele_frequency"],
                "gene_symbol": gene["gene_symbol"],
                "gene_start_position": gene["start_position"],
                "gene_end_position": gene["end_position"],
                "gene_biotype": gene["gene_biotype"],
                "overlap_length": overlap_length,
                "overlap_fraction_of_gene": overlap_fraction_of_gene,
                "overlap_fraction_of_deletion": overlap_fraction_of_deletion,
            })

    return pd.DataFrame(records)


def build_full_overlap_table(cnv, ensembl):
    all_chromosomes = sorted(cnv["chromosome"].unique())
    chunks = []

    for chrom in all_chromosomes:
        cnv_chr = cnv[cnv["chromosome"] == chrom]
        gene_chr = ensembl[ensembl["chromosome"] == chrom]

        if len(cnv_chr) == 0 or len(gene_chr) == 0:
            continue

        result = overlap_join_chromosome(cnv_chr, gene_chr)
        chunks.append(result)
        print(f"  chr{chrom}: {len(cnv_chr)} deletions, {len(gene_chr)} genes -> {len(result)} overlap pairs")

    return pd.concat(chunks, ignore_index=True)


def build_cnv_feature_matrix():
    cnv, ensembl, gnomad, clingen, omim = load_tables()

    print(f"Total deletions: {len(cnv)}")
    print(f"Total genes: {len(ensembl)}")
    print()
    print("Running per-chromosome overlap join...")

    df = build_full_overlap_table(cnv, ensembl)
    print(f"\nStep 1 - after overlap join: {len(df)} rows")

    df = df.merge(gnomad, on="gene_symbol", how="left")
    print(f"Step 2 - after gnomAD constraint join: {len(df)} rows")

    df = df.merge(clingen, on="gene_symbol", how="left")
    print(f"Step 3 - after ClinGen dosage join: {len(df)} rows")

    df = df.merge(omim, on="gene_symbol", how="left")
    print(f"Step 4 - after OMIM gene-disease-inheritance join: {len(df)} rows")

    return df


if __name__ == "__main__":
    feature_df = build_cnv_feature_matrix()

    print()
    print(f"Final CNV feature matrix: {len(feature_df)} rows, {len(feature_df.columns)} columns")
    print(f"Unique variant_id (deletions) count: {feature_df['variant_id'].nunique()}")
    print(f"Rows with no OMIM disease match: {feature_df['disease_name'].isna().sum()}")

    feature_df.to_csv("data/processed/cnv/cnv_features.csv", index=False)