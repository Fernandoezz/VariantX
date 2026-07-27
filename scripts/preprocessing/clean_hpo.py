import pandas as pd


# HPO's own frequency sub-ontology terms, mapped to the midpoint of their
# documented percentage range. Source: HPO frequency terms under HP:0040279.
HPO_FREQUENCY_MAP = {
    "HP:0040280": 1.00,   # Obligate (100%)
    "HP:0040281": 0.90,   # Very frequent (80-99%)
    "HP:0040282": 0.55,   # Frequent (30-79%)
    "HP:0040283": 0.17,   # Occasional (5-29%)
    "HP:0040284": 0.025,  # Very rare (1-4%)
    "HP:0040285": 0.00,   # Excluded (0%)
}


def parse_frequency(raw_freq):
    """
    Convert HPO's mixed frequency representations into a single numeric
    value between 0 and 1.

    Handles three formats:
      - HPO frequency term codes (e.g. 'HP:0040281') -> mapped via HPO_FREQUENCY_MAP
      - Fraction strings (e.g. '1/2', '3/3')          -> computed directly
      - Missing/unparseable                            -> left as NaN (not guessed)
    """
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

    df = df.rename(columns={"database_id": "disease_id"})

    df = df[[
        "disease_id", "disease_name", "hpo_id",
        "frequency", "frequency_numeric",
        "onset", "qualifier", "aspect"
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
    print(clean_df["frequency_numeric"].describe())

    clean_df.to_csv("data/interim/hpo_tables/disease_hpo_annotations_clean.csv", index=False)