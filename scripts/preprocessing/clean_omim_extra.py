import re
import pandas as pd


PHENOTYPE_PATTERN = re.compile(r"^(.*?),\s*(\d{6})\s*\((\d)\)(.*)$")


def parse_single_phenotype(entry):
    """Same parsing logic as clean_omim.py, reused for morbidmap's Phenotype column."""
    entry = entry.strip()
    if not entry:
        return None

    match = PHENOTYPE_PATTERN.match(entry)
    if not match:
        return {
            "disease_name": entry.strip("{}[]? "),
            "phenotype_mim_number": None,
            "mapping_key": None,
            "phenotype_type": "unconfirmed_or_unmapped",
            "inheritance_modes": []
        }

    raw_name, mim_number, mapping_key, remainder = match.groups()
    raw_name = raw_name.strip()

    if raw_name.startswith("{") and raw_name.endswith("}"):
        phenotype_type = "susceptibility"
    elif raw_name.startswith("[") and raw_name.endswith("]"):
        phenotype_type = "nondisease"
    elif raw_name.startswith("?"):
        phenotype_type = "unconfirmed_mendelian"
    else:
        phenotype_type = "confirmed_mendelian"

    disease_name = raw_name.strip("{}[]? ")
    inheritance_modes = [m.strip() for m in remainder.split(",") if m.strip()]

    return {
        "disease_name": disease_name,
        "phenotype_mim_number": mim_number,
        "mapping_key": mapping_key,
        "phenotype_type": phenotype_type,
        "inheritance_modes": inheritance_modes
    }


def clean_mim2gene(df):
    df = df.copy()
    df = df.dropna(subset=["gene_symbol"])
    return df


def clean_mim_titles(df):
    df = df.copy()

    # Preferred title format: "DISEASE NAME; SYMBOL" - split off the symbol if present
    split_titles = df["preferred_title_raw"].str.split(";", n=1, expand=True)
    df["preferred_disease_name"] = split_titles[0].str.strip()
    df["preferred_symbol"] = split_titles[1].str.strip() if split_titles.shape[1] > 1 else None

    df = df[["mim_number", "prefix", "preferred_disease_name", "preferred_symbol"]]
    return df


def clean_morbidmap(df):
    df = df.copy()
    df = df.dropna(subset=["gene_symbols_raw"])

    records = []
    for _, row in df.iterrows():
        parsed = parse_single_phenotype(row["phenotype_raw"])
        if parsed is None:
            continue

        # First symbol listed is OMIM's approved gene symbol; the rest
        # are historical aliases, not distinct genes.
        gene_symbol = row["gene_symbols_raw"].split(",")[0].strip()

        records.append({
            "gene_symbol": gene_symbol,
            "gene_mim_number": row["gene_mim_number"],
            "disease_name": parsed["disease_name"],
            "phenotype_mim_number": parsed["phenotype_mim_number"],
            "mapping_key": parsed["mapping_key"],
            "phenotype_type": parsed["phenotype_type"],
            "inheritance_modes": ";".join(parsed["inheritance_modes"]) if parsed["inheritance_modes"] else None,
            "cyto_location": row["cyto_location"],
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    mim2gene_raw = pd.read_csv("data/interim/omim_tables/mim2gene_raw.csv", dtype=str)
    mim2gene_clean = clean_mim2gene(mim2gene_raw)
    print(f"mim2gene: {len(mim2gene_raw)} -> {len(mim2gene_clean)} rows (dropped rows with no gene_symbol)")
    mim2gene_clean.to_csv("data/interim/omim_tables/mim2gene_clean.csv", index=False)

    print()
    titles_raw = pd.read_csv("data/interim/omim_tables/mim_titles_raw.csv", dtype=str)
    titles_clean = clean_mim_titles(titles_raw)
    print(f"mimTitles: {len(titles_raw)} -> {len(titles_clean)} rows")
    titles_clean.to_csv("data/interim/omim_tables/mim_titles_clean.csv", index=False)

    print()
    morbidmap_raw = pd.read_csv("data/interim/omim_tables/morbidmap_raw.csv", dtype=str)
    morbidmap_clean = clean_morbidmap(morbidmap_raw)
    print(f"morbidmap: {len(morbidmap_raw)} raw rows -> {len(morbidmap_clean)} gene-disease pairs")
    print(f"Unique genes in morbidmap: {morbidmap_clean['gene_symbol'].nunique()}")
    print(morbidmap_clean["phenotype_type"].value_counts())
    morbidmap_clean.to_csv("data/interim/omim_tables/morbidmap_clean.csv", index=False)