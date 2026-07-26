import gzip
import pandas as pd


def parse_info(info_string):
    """Parse a VCF INFO field string into a dict."""
    info_dict = {}
    for item in info_string.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            info_dict[key] = value
    return info_dict


def read_clinvar_vcf(filepath):
    """Read ClinVar VCF (gzipped) into a raw DataFrame. No filtering, no row cap."""
    rows = []

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            info_dict = parse_info(fields[7])

            rows.append({
                "chromosome": fields[0],
                "position": int(fields[1]),
                "reference_allele": fields[3],
                "alternate_allele": fields[4],
                "gene_info_raw": info_dict.get("GENEINFO"),
                "clinvar_significance_raw": info_dict.get("CLNSIG"),
                "review_status_raw": info_dict.get("CLNREVSTAT"),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = read_clinvar_vcf("data/raw/clinvar/clinvar.vcf.gz")
    print(f"Read {len(df)} raw ClinVar records")
    df.to_csv("data/interim/clinvar_tables/clinvar_variants_raw.csv", index=False)