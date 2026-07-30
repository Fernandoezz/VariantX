import re
import pandas as pd


# Anchors on "123456 (2)" - a MIM number followed by its mapping key in
# parentheses. This is the one fixed, unambiguous pattern in an otherwise
# free-text field, so everything is parsed relative to it.
PHENOTYPE_PATTERN = re.compile(r"^(.*?),\s*(\d{6})\s*\((\d)\)(.*)$")


def parse_single_phenotype(entry):
    """
    Parses one phenotype entry, e.g.:
    '{Melanoma, cutaneous malignant, 1}, 155600 (2), Autosomal dominant'
    into disease name, MIM number, mapping key, phenotype type, and a
    list of inheritance modes.
    """
    entry = entry.strip()
    if not entry:
        return None

    match = PHENOTYPE_PATTERN.match(entry)
    if not match:
        # No MIM number found - keep the raw text, everything else blank
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

    inheritance_modes = [
        mode.strip() for mode in remainder.split(",") if mode.strip()
    ]

    return {
        "disease_name": disease_name,
        "phenotype_mim_number": mim_number,
        "mapping_key": mapping_key,
        "phenotype_type": phenotype_type,
        "inheritance_modes": inheritance_modes
    }


def expand_phenotypes(df):
    """
    Each gene row can list multiple phenotypes separated by ';'.
    Explodes into one row per gene-disease pair.
    """
    records = []

    for _, row in df.iterrows():
        if pd.isna(row["phenotypes_raw"]) or pd.isna(row["approved_gene_symbol"]):
            continue

        entries = row["phenotypes_raw"].split(";")

        for entry in entries:
            parsed = parse_single_phenotype(entry)
            if parsed is None:
                continue

            records.append({
                "gene_symbol": row["approved_gene_symbol"],
                "mim_number": row["mim_number"],
                "disease_name": parsed["disease_name"],
                "phenotype_mim_number": parsed["phenotype_mim_number"],
                "mapping_key": parsed["mapping_key"],
                "phenotype_type": parsed["phenotype_type"],
                "inheritance_modes": ";".join(parsed["inheritance_modes"]) if parsed["inheritance_modes"] else None,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    raw_df = pd.read_csv("data/interim/omim_tables/genemap2_raw.csv", dtype=str)

    clean_df = expand_phenotypes(raw_df)

    print(f"Gene-level rows in raw file: {len(raw_df)}")
    print(f"Gene-disease pairs after expansion: {len(clean_df)}")
    print(f"Unique genes with at least one disease: {clean_df['gene_symbol'].nunique()}")
    print()
    print("phenotype_type distribution:")
    print(clean_df["phenotype_type"].value_counts())
    print()
    print(f"Rows with inheritance_modes present: {clean_df['inheritance_modes'].notna().sum()}")
    print(f"Rows with NO inheritance_modes listed: {clean_df['inheritance_modes'].isna().sum()}")

    clean_df.to_csv("data/interim/omim_tables/gene_disease_inheritance.csv", index=False)