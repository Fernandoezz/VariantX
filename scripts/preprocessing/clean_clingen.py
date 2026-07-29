import pandas as pd


HAPLOINSUFFICIENCY_MAP = {
    "Sufficient Evidence for Haploinsufficiency": 3,
    "Some Evidence for Haploinsufficiency": 2,
    "Emerging Evidence for Haploinsufficiency": 1,
    "Little Evidence for Haploinsufficiency": 1,
    "No Evidence for Haploinsufficiency": 0,
    "Dosage Sensitivity Unlikely for Haploinsufficiency": 0,
    "Gene Associated with Autosomal Recessive Phenotype": 0,
    "Not yet evaluated": None,
}

TRIPLOSENSITIVITY_MAP = {
    "Sufficient Evidence for Triplosensitivity": 3,
    "Some Evidence for Triplosensitivity": 2,
    "Emerging Evidence for Triplosensitivity": 1,
    "Little Evidence for Triplosensitivity": 1,
    "No Evidence for Triplosensitivity": 0,
    "Dosage Sensitivity Unlikely for Triplosensitivity": 0,
    "Gene Associated with Autosomal Recessive Phenotype": 0,
    "Not yet evaluated": None,
}


def is_junk_separator_row(df):
    """
    Detects ClinGen's junk separator row, where label columns contain a run
    of plus signs instead of a real label (length varies between file
    versions, so match on pattern, not exact string).
    """
    is_junk = (
        df["HAPLOINSUFFICIENCY"].astype(str).str.fullmatch(r"\++", na=False) |
        df["TRIPLOSENSITIVITY"].astype(str).str.fullmatch(r"\++", na=False) |
        df["GENE SYMBOL"].astype(str).str.fullmatch(r"\++", na=False)
    )
    return is_junk


def clean_clingen_dosage(df):
    df = df.copy()

    junk_mask = is_junk_separator_row(df)
    df = df[~junk_mask]
    df = df.reset_index(drop=True)

    df = df[["GENE SYMBOL", "HAPLOINSUFFICIENCY", "TRIPLOSENSITIVITY"]]
    df.columns = ["gene_symbol", "haploinsufficiency_label", "triplosensitivity_label"]

    df["haploinsufficiency_score"] = df["haploinsufficiency_label"].map(HAPLOINSUFFICIENCY_MAP)
    df["triplosensitivity_score"] = df["triplosensitivity_label"].map(TRIPLOSENSITIVITY_MAP)

    unmapped_haplo = df["haploinsufficiency_label"].notna() & df["haploinsufficiency_score"].isna() & (df["haploinsufficiency_label"] != "Not yet evaluated")
    unmapped_triplo = df["triplosensitivity_label"].notna() & df["triplosensitivity_score"].isna() & (df["triplosensitivity_label"] != "Not yet evaluated")

    df = df[[
        "gene_symbol",
        "haploinsufficiency_label", "haploinsufficiency_score",
        "triplosensitivity_label", "triplosensitivity_score"
    ]]

    return df, unmapped_haplo.sum(), unmapped_triplo.sum(), junk_mask.sum()


if __name__ == "__main__":
    raw_df = pd.read_csv("data/interim/clingen_tables/gene_dosage_raw.csv")

    clean_df, unmapped_haplo, unmapped_triplo, junk_rows_removed = clean_clingen_dosage(raw_df)

    print(f"Raw rows: {len(raw_df)} -> Clean rows: {len(clean_df)}")
    print(f"Junk separator rows removed: {junk_rows_removed}")
    print(f"Unmapped haploinsufficiency labels (should be 0): {unmapped_haplo}")
    print(f"Unmapped triplosensitivity labels (should be 0): {unmapped_triplo}")
    print()
    print("Haploinsufficiency score distribution:")
    print(clean_df["haploinsufficiency_score"].value_counts(dropna=False))
    print()
    print("Triplosensitivity score distribution:")
    print(clean_df["triplosensitivity_score"].value_counts(dropna=False))

    clean_df.to_csv("data/interim/clingen_tables/gene_dosage_clean.csv", index=False)