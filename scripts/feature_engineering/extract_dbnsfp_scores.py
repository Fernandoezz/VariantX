import gzip
import pandas as pd


DBNSFP_COLUMNS_NEEDED = {
    0: "chr", 1: "position", 2: "ref", 3: "alt",
    48: "SIFT_score", 49: "SIFT_converted_rankscore", 50: "SIFT_pred",
    54: "Polyphen2_HDIV_score", 55: "Polyphen2_HDIV_rankscore", 56: "Polyphen2_HDIV_pred",
    57: "Polyphen2_HVAR_score", 58: "Polyphen2_HVAR_rankscore", 59: "Polyphen2_HVAR_pred",
    152: "CADD_raw", 153: "CADD_raw_rankscore", 154: "CADD_phred",
    166: "GERP_NR", 167: "GERP_RS",
    171: "phyloP100way_vertebrate", 172: "phyloP100way_vertebrate_rankscore",
    177: "phastCons100way_vertebrate", 178: "phastCons100way_vertebrate_rankscore",
}


def build_needed_variant_lookup(variant_ids):
    """
    variant_id format: 'chromosome_position_ref_alt'
    Returns a dict keyed by (chr, position, ref, alt) tuple for fast matching
    during the single streaming pass.
    """
    lookup = {}
    for vid in variant_ids:
        parts = vid.split("_")
        if len(parts) != 4:
            continue
        chrom, pos, ref, alt = parts
        lookup[(chrom, pos, ref, alt)] = vid
    return lookup


def extract_matching_rows(filepath, needed_lookup):
    """
    Single sequential pass through dbNSFP. For every line whose
    (chr, pos, ref, alt) matches a variant we need, extract the needed
    columns. Does not require sorted input matching or random access -
    just checks every line against the lookup dict (O(1) per line).
    """
    max_index_needed = max(DBNSFP_COLUMNS_NEEDED.keys())
    records = []
    lines_read = 0
    matches_found = 0

    with gzip.open(filepath, "rt") as f:
        header = f.readline()  # skip header

        for line in f:
            lines_read += 1
            if lines_read % 5_000_000 == 0:
                print(f"  ...scanned {lines_read:,} lines, {matches_found} matches found so far")

            fields = line.rstrip("\n").split("\t", max_index_needed + 1)

            key = (fields[0], fields[1], fields[2], fields[3])
            if key not in needed_lookup:
                continue

            matches_found += 1
            row = {"variant_id": needed_lookup[key]}
            for idx, colname in DBNSFP_COLUMNS_NEEDED.items():
                if idx in (0, 1, 2, 3):
                    continue
                value = fields[idx] if idx < len(fields) else "."
                row[colname] = None if value == "." else value

            records.append(row)

    print(f"Total lines scanned: {lines_read:,}")
    print(f"Total matches found: {matches_found:,}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Loading variant IDs needed from SNV training table...")
    training_df = pd.read_csv("data/processed/snv/snv_training_table.csv", dtype={"chromosome": str})
    unique_variant_ids = training_df["variant_id"].unique().tolist()
    print(f"Unique variants needed: {len(unique_variant_ids)}")

    needed_lookup = build_needed_variant_lookup(unique_variant_ids)
    print(f"Lookup dict built: {len(needed_lookup)} entries")

    print("Starting single-pass scan of dbNSFP (this will take a while - 46.8GB file)...")
    dbnsfp_scores = extract_matching_rows(
        "data/raw/dbnsfp/dbNSFP5.3.1a_grch38.gz", needed_lookup
    )

    print(f"Extracted {len(dbnsfp_scores)} rows")

    dbnsfp_scores.to_csv("data/interim/dbnsfp_tables/dbnsfp_scores_raw.csv", index=False)