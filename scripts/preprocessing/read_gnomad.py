import pandas as pd


def read_gnomad_constraint(filepath):
    """
    Read the gnomAD gene constraint file (LOF metrics by gene).
    Raw load only — no filtering or renaming here.
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        compression="gzip"
    )
    return df


if __name__ == "__main__":
    df = read_gnomad_constraint("data/raw/gnomad/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz")
    print(f"Read {len(df)} rows, {len(df.columns)} columns")

    df.to_csv("data/interim/gnomad_tables/gene_constraint_raw.csv", index=False)