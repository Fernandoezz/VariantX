import pandas as pd


HPO_FREQUENCY_MAP = {
    "HP:0040280": 1.00,
    "HP:0040281": 0.90,
    "HP:0040282": 0.55,
    "HP:0040283": 0.17,
    "HP:0040284": 0.025,
    "HP:0040285": 0.00,
}


def parse_frequency(raw_freq):
    if pd.isna(raw_freq):
        return None

    raw_freq = str(raw_freq).strip()

    if raw_freq in HPO_FREQUENCY_MAP:
        return HPO_FREQUENCY_MAP[raw_freq]

    if "/" in raw_freq:
        try:
            numerator, denominator = raw_freq.split("/")
            numerator, denominator = float(numerator), float(denominator)
            if denominator == 0:
                return None
            return numerator / denominator
        except ValueError:
            return None

    return None


def clean_hpo_annotations(df):
    df = df.copy()

    df["frequency_numeric"] = df["frequency"].apply(parse_frequency)

    # qualifier holds "NOT" when a disease explicitly does NOT show a symptom.
    # Flag it so downstream similarity scoring doesn't treat negated rows
    # as positive evidence.
    df["is_negated"] = df["qualifier"].astype(str).str.strip().str.upper() == "NOT"

    df = df.rename(columns={"database_id": "disease_id"})

    df = df[[
        "disease_id", "disease_name", "hpo_id",
        "frequency", "frequency_numeric",
        "onset", "qualifier", "is_negated", "aspect"
    ]]

    return df


if __name__ == "__main__":
    raw_df = pd.read_csv(
        "data/interim/hpo_tables/disease_hpo_annotations_raw.csv",
        dtype=str
    )
    clean_df = clean_hpo_annotations(raw_df)

    print(f"Rows: {len(clean_df)}")
    print(f"Rows with usable frequency_numeric: {clean_df['frequency_numeric'].notna().sum()}")
    print(f"Rows with missing frequency: {clean_df['frequency_numeric'].isna().sum()}")
    print(f"Rows flagged is_negated=True: {clean_df['is_negated'].sum()}")
    print(clean_df["aspect"].value_counts(dropna=False))

    clean_df.to_csv("data/interim/hpo_tables/disease_hpo_annotations_clean.csv", index=False)