import pandas as pd

HAPLOINSUFFICIENCY_MAP = {
    "Sufficient Evidence for Haploinsufficiency": 3,
    "Some Evidence for Haploinsufficiency": 2,
    "Emerging Evidence for Haploinsufficiency": 1,
    "Little Evidence for Haploinsufficiency": 1,
    "No Evidence for Haploinsufficiency": 0,
    "Dosage Sensitivity Unlikely for Haploinsufficiency": 0,
    "Gene Associated with Autosomal Recessive Phenotype": 0,
    "Not yet evaluated": None,
}

TRIPLOSENSITIVITY_MAP = {
    "Sufficient Evidence for Triplosensitivity": 3,
    "Some Evidence for Triplosensitivity": 2,
    "Emerging Evidence for Triplosensitivity": 1,
    "Little Evidence for Triplosensitivity": 1,
    "No Evidence for Triplosensitivity": 0,
    "Dosage Sensitivity Unlikely for Triplosensitivity": 0,
    "Gene Associated with Autosomal Recessive Phenotype": 0,
    "Not yet evaluated": None,
}

df = pd.read_csv("data/interim/clingen_tables/gene_dosage_raw.csv")

haplo_unmapped = df[~df["HAPLOINSUFFICIENCY"].isin(HAPLOINSUFFICIENCY_MAP.keys())]
triplo_unmapped = df[~df["TRIPLOSENSITIVITY"].isin(TRIPLOSENSITIVITY_MAP.keys())]

print("Unmapped HAPLOINSUFFICIENCY values:")
print(haplo_unmapped["HAPLOINSUFFICIENCY"].unique())

print("Unmapped TRIPLOSENSITIVITY values:")
print(triplo_unmapped["TRIPLOSENSITIVITY"].unique())