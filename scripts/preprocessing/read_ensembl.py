import gzip
import pandas as pd


def parse_attributes(attribute_string):
    """
    GTF attribute field looks like:
    'gene_id "ENSG00000223972"; gene_version "5"; gene_name "DDX11L1"; gene_biotype "lncRNA";'
    Returns a dict of key -> value.
    """
    attr_dict = {}
    for item in attribute_string.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        if " " in item:
            key, value = item.split(" ", 1)
            attr_dict[key] = value.strip('"')
    return attr_dict


def read_ensembl_genes(filepath):
    """
    Read Ensembl GTF, keeping only 'gene' feature rows (not exons/transcripts/CDS).
    Raw load only - no null-dropping or dedup here.
    """
    rows = []

    with gzip.open(filepath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")

            if fields[2] != "gene":
                continue

            attrs = parse_attributes(fields[8])

            rows.append({
                "chromosome": fields[0],
                "start_position": int(fields[3]),
                "end_position": int(fields[4]),
                "gene_symbol": attrs.get("gene_name"),
                "gene_id": attrs.get("gene_id"),
                "gene_biotype": attrs.get("gene_biotype"),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = read_ensembl_genes("data/raw/ensembl/Homo_sapiens.GRCh38.116.gtf.gz")
    print(f"Read {len(df)} gene rows")
    print(f"Missing gene_symbol: {df['gene_symbol'].isna().sum()}")
    print(f"Unique chromosome values: {sorted(df['chromosome'].unique())}")

    df.to_csv("data/interim/ensembl_tables/gene_coordinates_raw.csv", index=False)