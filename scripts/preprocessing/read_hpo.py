import pronto
import pandas as pd

# Load ontology
ontology = pronto.Ontology("data/raw/hpo/hp.obo")

rows = []

for term in ontology.terms():
    rows.append({
        "HPO_ID": term.id,
        "Name": term.name,
        "Definition": term.definition,
        "Parents": ";".join([parent.id for parent in term.superclasses(distance=1)])
    })

df = pd.DataFrame(rows)

print(df.head())

# Save as csv
df.to_csv(
    "data/interim/hpo_tables/hpo_terms.csv",
    index=False
)