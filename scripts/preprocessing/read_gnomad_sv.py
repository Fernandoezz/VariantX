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


def read_gnomad_sv(filepath):
    """
    Read gnomAD-SV VCF (gzipped) into a raw DataFrame.
    No filtering here - keeps all SV types and all FILTER statuses.
    """
    rows = []

    with gzip.open(filepath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            info = parse_info(fields[7])

            rows.append({
                "chromosome": fields[0],
                "start_position": int(fields[1]),
                "variant_filter": fields[6],
                "sv_type_raw": info.get("SVTYPE"),
                "end_position_raw": info.get("END"),
                "sv_length_raw": info.get("SVLEN"),
                "allele_frequency_raw": info.get("AF"),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = read_gnomad_sv("data/raw/gnomad_sv/gnomad.v4.1.sv.sites.vcf.gz")
    print(f"Read {len(df)} raw SV records")
    print(f"sv_type_raw distribution:")
    print(df["sv_type_raw"].value_counts(dropna=False))
    print(f"variant_filter distribution:")
    print(df["variant_filter"].value_counts(dropna=False))

    df.to_csv("data/interim/gnomad_sv_tables/gnomad_sv_raw.csv", index=False)