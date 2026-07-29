import pandas as pd


CONSTRAINT_COLUMNS = [
    "gene", "oe_lof", "oe_lof_lower", "oe_lof_upper", "oe_lof_upper_rank", "pLI"
]


def clean_gnomad_constraint(df):
    df = df.copy()

    df = df[CONSTRAINT_COLUMNS]
    df = df.rename(columns={"gene": "gene_symbol"})

    before = len(df)

    # Some genes have multiple rows (multi-transcript entries) with genuinely
    # different constraint estimates - not exact duplicates. Keep the row with
    # the lowest oe_lof_upper_rank (gnomAD's own confidence ranking; lower =
    # more reliable), breaking ties by keeping the first occurrence.
    df = df.sort_values("oe_lof_upper_rank", na_position="last")
    df = df.drop_duplicates(subset=["gene_symbol"], keep="first")
    df = df.sort_index()  # restore original row order

    duplicates_dropped = before - len(df)

    return df, duplicates_dropped


if __name__ == "__main__":
    raw_df = pd.read_csv("data/interim/gnomad_tables/gene_constraint_raw.csv")

    clean_df, duplicates_dropped = clean_gnomad_constraint(raw_df)

    print(f"Raw rows: {len(raw_df)} -> Clean rows: {len(clean_df)}")
    print(f"Duplicate gene_symbol rows resolved (kept most reliable): {duplicates_dropped}")
    print(f"Missing oe_lof: {clean_df['oe_lof'].isna().sum()}")
    print(f"Missing pLI: {clean_df['pLI'].isna().sum()}")
    print(clean_df[["oe_lof", "pLI"]].describe())

    clean_df.to_csv("data/interim/gnomad_tables/gene_constraint_clean.csv", index=False)