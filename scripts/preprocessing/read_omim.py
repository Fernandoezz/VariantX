import pandas as pd


OMIM_COLUMNS = [
    "chromosome_raw", "genomic_start", "genomic_end", "cyto_location",
    "computed_cyto_location", "mim_number", "gene_symbols_raw", "gene_name",
    "approved_gene_symbol", "entrez_gene_id", "ensembl_gene_id",
    "comments", "phenotypes_raw", "mouse_gene_symbol"
]


def read_genemap2(filepath):
    """
    Read OMIM's genemap2.txt.
    First 3 lines are pure comments; line 4 is the real header (still
    '#'-prefixed, so we skip it via skiprows and supply column names
    manually rather than letting pandas parse it as a comment).
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        skiprows=4,
        names=OMIM_COLUMNS,
        dtype=str
    )
    return df


if __name__ == "__main__":
    df = read_genemap2("data/raw/omim/genemap2.txt")
    print(f"Read {len(df)} rows")
    print(f"Rows with a non-empty approved_gene_symbol: {df['approved_gene_symbol'].notna().sum()}")
    print(f"Rows with non-empty phenotypes_raw: {df['phenotypes_raw'].notna().sum()}")

    df.to_csv("data/interim/omim_tables/genemap2_raw.csv", index=False)