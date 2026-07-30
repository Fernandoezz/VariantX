import pandas as pd

training_df = pd.read_csv("data/processed/snv/snv_training_table.csv", dtype={"chromosome": str})
dbnsfp_df = pd.read_csv("data/interim/dbnsfp_tables/dbnsfp_scores_raw.csv")

dbnsfp_df = dbnsfp_df.drop_duplicates(subset=["variant_id"], keep="first")

before = len(training_df)
merged = training_df.merge(dbnsfp_df, on="variant_id", how="left")
after = len(merged)

print(f"Training table rows before: {before}, after dbNSFP join: {after}")
print(f"Rows with SIFT score present: {merged['SIFT_score'].notna().sum()}")
print(f"Rows with CADD_phred present: {merged['CADD_phred'].notna().sum()}")

merged.to_csv("data/processed/snv/snv_training_table_final.csv", index=False)