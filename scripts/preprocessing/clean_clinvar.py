import pandas as pd


def parse_gene_info(gene_info_raw):
    """
    GENEINFO looks like 'OR4F5:79501' or 'GENE1:ID1|GENE2:ID2'
    for variants overlapping multiple genes.
    Returns a list of all gene symbols (not just the first).
    """
    if pd.isna(gene_info_raw):
        return []

    entries = gene_info_raw.split("|")
    gene_symbols = [entry.split(":")[0] for entry in entries]
    return gene_symbols


def simplify_significance(raw_sig):
    """
    Collapse ClinVar's ~100 raw CLNSIG combinations into a small controlled set.
    Splits compound values (e.g. 'Pathogenic/Likely_pathogenic') into individual
    assertion tokens and matches them exactly, so 'likely_pathogenic' is never
    confused with a substring of 'pathogenic'. When a variant has mixed/compound
    classifications, this biases toward the stronger label (Pathogenic over
    Likely_pathogenic, Benign over Likely_benign).
    """
    if pd.isna(raw_sig):
        return None

    sig = raw_sig.lower()

    if "conflicting" in sig:
        return "Conflicting"

    tokens = sig.replace("/", "|").split("|")

    if any(t == "pathogenic" for t in tokens):
        return "Pathogenic"
    if any(t == "likely_pathogenic" for t in tokens):
        return "Likely_pathogenic"
    if any(t == "benign" for t in tokens):
        return "Benign"
    if any(t == "likely_benign" for t in tokens):
        return "Likely_benign"
    if "uncertain" in sig:
        return "Uncertain_significance"

    return "Other"  # risk_factor, drug_response, association, protective, etc.


def clean_clinvar(df):
    df = df.copy()

    df["clinvar_significance"] = df["clinvar_significance_raw"].apply(simplify_significance)
    df = df.rename(columns={"review_status_raw": "review_status"})

    # Drop variants with no clinical significance
    df = df.dropna(subset=["clinvar_significance"])

    # Stable identity for the physical variant, computed BEFORE exploding,
    # so both exploded copies (one per overlapping gene) carry the same ID.
    # Used later for grouped train/test splitting to avoid leakage.
    df["variant_id"] = (
        df["chromosome"].astype(str) + "_" +
        df["position"].astype(str) + "_" +
        df["reference_allele"].astype(str) + "_" +
        df["alternate_allele"].astype(str)
    )

    # Parse gene list, flag multi-gene variants, then explode into one row per gene
    df["gene_symbol_list"] = df["gene_info_raw"].apply(parse_gene_info)
    df["has_multiple_genes"] = df["gene_symbol_list"].apply(lambda g: len(g) > 1)

    df = df.explode("gene_symbol_list")
    df = df.rename(columns={"gene_symbol_list": "gene_symbol"})

    # Drop rows where no gene could be resolved at all
    df = df.dropna(subset=["gene_symbol"])
    df = df[df["gene_symbol"] != ""]

    df = df[[
        "variant_id", "chromosome", "position", "reference_allele", "alternate_allele",
        "gene_symbol", "has_multiple_genes",
        "clinvar_significance", "review_status"
    ]]

    return df


if __name__ == "__main__":
    raw_df = pd.read_csv(
        "data/interim/clinvar_tables/clinvar_variants_raw.csv",
        dtype={"chromosome": str}
    )
    clean_df = clean_clinvar(raw_df)

    print(f"Raw rows: {len(raw_df)} -> Clean rows: {len(clean_df)}")
    print(f"Unique variant_id count: {clean_df['variant_id'].nunique()}")
    print(f"Rows flagged has_multiple_genes=True: {clean_df['has_multiple_genes'].sum()}")
    print(clean_df["clinvar_significance"].value_counts())

    clean_df.to_csv("data/interim/clinvar_tables/clinvar_variants_clean.csv", index=False)