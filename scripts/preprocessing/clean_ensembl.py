import pandas as pd


STANDARD_CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]


def clean_ensembl_genes(df):
    df = df.copy()

    before = len(df)
    df = df.dropna(subset=["gene_symbol"])
    dropped_no_symbol = before - len(df)

    before_contig_filter = len(df)
    df = df[df["chromosome"].isin(STANDARD_CHROMOSOMES)]
    dropped_alt_contigs = before_contig_filter - len(df)

    # NOTE: gene_symbol is intentionally NOT deduplicated here. Some genes
    # (e.g. multi-copy rRNA/snRNA genes like 5S_rRNA, and pseudoautosomal
    # genes shared between X/Y) legitimately exist at more than one real
    # genomic location. Downstream CNV-overlap joins should match on
    # chromosome + coordinate range, not on gene_symbol alone.
    duplicate_symbols = df[df.duplicated(subset=["gene_symbol"], keep=False)]

    df = df.reset_index(drop=True)

    return df, dropped_no_symbol, dropped_alt_contigs, duplicate_symbols


if __name__ == "__main__":
    raw_df = pd.read_csv("data/interim/ensembl_tables/gene_coordinates_raw.csv")

    clean_df, dropped_no_symbol, dropped_alt_contigs, duplicate_symbols = clean_ensembl_genes(raw_df)

    print(f"Raw rows: {len(raw_df)} -> Clean rows: {len(clean_df)}")
    print(f"Dropped (no gene_symbol): {dropped_no_symbol}")
    print(f"Dropped (non-standard chromosome/contig): {dropped_alt_contigs}")
    print(f"Gene symbols with multiple genomic loci (kept, not deduped): {duplicate_symbols['gene_symbol'].nunique()}")
    print(f"Total rows involved in multi-locus symbols: {len(duplicate_symbols)}")

    clean_df.to_csv("data/interim/ensembl_tables/gene_coordinates_clean.csv", index=False)