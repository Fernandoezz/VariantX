import pandas as pd

df = pd.read_csv("data/processed/snv/snv_training_table_final.csv", dtype={"chromosome": str}, low_memory=False)

print(f"Total rows: {len(df)}")
print(f"Unique variant_ids: {df['variant_id'].nunique()}")
print()

# Row-level coverage
print(f"Rows with SIFT_score present: {df['SIFT_score'].notna().sum()} ({100*df['SIFT_score'].notna().sum()/len(df):.1f}%)")
print(f"Rows with CADD_phred present: {df['CADD_phred'].notna().sum()} ({100*df['CADD_phred'].notna().sum()/len(df):.1f}%)")

# Unique-variant-level coverage (deduplicated)
unique_df = df.drop_duplicates(subset=["variant_id"])
print()
print(f"Unique variants with SIFT_score: {unique_df['SIFT_score'].notna().sum()} / {len(unique_df)} ({100*unique_df['SIFT_score'].notna().sum()/len(unique_df):.1f}%)")

# Coverage broken down by causal vs background
print()
print("=== Coverage by row type ===")
causal_df = df[df["is_causal"] == 1]
background_df = df[df["is_causal"] == 0]

print(f"Causal rows: {len(causal_df)}, with SIFT: {causal_df['SIFT_score'].notna().sum()} ({100*causal_df['SIFT_score'].notna().sum()/len(causal_df):.1f}%)")
print(f"Background rows: {len(background_df)}, with SIFT: {background_df['SIFT_score'].notna().sum()} ({100*background_df['SIFT_score'].notna().sum()/len(background_df):.1f}%)")

# Check consequence type context - is missing SIFT correlated with variant type?
print()
print("=== clinvar_significance breakdown for rows MISSING SIFT ===")
missing_sift = df[df["SIFT_score"].isna()]
print(missing_sift["clinvar_significance"].value_counts())