import pandas as pd


def read_mim2gene(filepath):
    df = pd.read_csv(
        filepath, sep="\t", skiprows=5,
        names=["mim_number", "mim_entry_type", "entrez_gene_id", "gene_symbol", "ensembl_gene_id"],
        dtype=str
    )
    return df


def read_mim_titles(filepath):
    df = pd.read_csv(
        filepath, sep="\t", skiprows=3,
        names=["prefix", "mim_number", "preferred_title_raw", "alternative_titles_raw", "included_titles_raw"],
        dtype=str
    )
    return df


def read_morbidmap(filepath):
    df = pd.read_csv(
        filepath, sep="\t", skiprows=4,
        names=["phenotype_raw", "gene_symbols_raw", "gene_mim_number", "cyto_location"],
        dtype=str
    )
    return df


if __name__ == "__main__":
    mim2gene_df = read_mim2gene("data/raw/omim/mim2gene.txt")
    print(f"mim2gene: {len(mim2gene_df)} rows")
    print(mim2gene_df["mim_entry_type"].value_counts())
    mim2gene_df.to_csv("data/interim/omim_tables/mim2gene_raw.csv", index=False)

    print()
    titles_df = read_mim_titles("data/raw/omim/mimTitles.txt")
    print(f"mimTitles: {len(titles_df)} rows")
    titles_df.to_csv("data/interim/omim_tables/mim_titles_raw.csv", index=False)

    print()
    morbidmap_df = read_morbidmap("data/raw/omim/morbidmap.txt")
    print(f"morbidmap: {len(morbidmap_df)} rows")
    morbidmap_df.to_csv("data/interim/omim_tables/morbidmap_raw.csv", index=False)