import pandas as pd


def clean_gnomad_sv(df):
    df = df.copy()

    df = df[df["sv_type_raw"] == "DEL"]

    before_filter = len(df)
    df = df[df["variant_filter"] == "PASS"]
    dropped_low_quality = before_filter - len(df)

    # Normalize chromosome naming to match Ensembl/ClinVar convention
    # (gnomAD-SV uses 'chr21' style, Ensembl uses plain '21')
    df["chromosome"] = df["chromosome"].str.replace("^chr", "", regex=True)

    df["end_position"] = pd.to_numeric(df["end_position_raw"], errors="coerce")
    df["allele_frequency"] = pd.to_numeric(df["allele_frequency_raw"], errors="coerce")
    df["deletion_size"] = pd.to_numeric(df["sv_length_raw"], errors="coerce").abs()
    df["deletion_size_from_coords"] = df["end_position"] - df["start_position"]

    missing_end = df["end_position"].isna().sum()
    missing_af = df["allele_frequency"].isna().sum()
    missing_svlen = df["deletion_size"].isna().sum()

    df["variant_id"] = (
        df["chromosome"].astype(str) + "_" +
        df["start_position"].astype(str) + "_" +
        df["end_position"].astype(str)
    )

    df = df[[
        "variant_id", "chromosome", "start_position", "end_position",
        "deletion_size", "deletion_size_from_coords",
        "allele_frequency"
    ]]

    return df, dropped_low_quality, missing_end, missing_af, missing_svlen


if __name__ == "__main__":
    raw_df = pd.read_csv("data/interim/gnomad_sv_tables/gnomad_sv_raw.csv")

    clean_df, dropped_low_quality, missing_end, missing_af, missing_svlen = clean_gnomad_sv(raw_df)

    print(f"Raw rows: {len(raw_df)} -> Clean rows (DEL, PASS only): {len(clean_df)}")
    print(f"Dropped for failing quality filter: {dropped_low_quality}")
    print(f"Missing end_position: {missing_end}")
    print(f"Missing allele_frequency: {missing_af}")
    print(f"Missing deletion_size (SVLEN): {missing_svlen}")
    print(clean_df[["deletion_size", "allele_frequency"]].describe())

    clean_df.to_csv("data/interim/gnomad_sv_tables/population_frequency_clean.csv", index=False)