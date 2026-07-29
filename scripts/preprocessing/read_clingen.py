import pandas as pd


def read_clingen_dosage(filepath):
    """
    Read the ClinGen Dosage Sensitivity CSV.
    The file has 4 metadata/header lines before the real column header,
    hence skiprows=4.
    Raw load only - no filtering or score mapping here.
    """
    df = pd.read_csv(filepath, skiprows=4)
    return df


if __name__ == "__main__":
    df = read_clingen_dosage("data/raw/clingen/ClinGen-Dosage-Sensitivity-2026-06-11.csv")
    print(f"Read {len(df)} rows, {len(df.columns)} columns")
    print(df.columns.tolist())

    df.to_csv("data/interim/clingen_tables/gene_dosage_raw.csv", index=False)