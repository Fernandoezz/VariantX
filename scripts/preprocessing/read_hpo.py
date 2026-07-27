import pandas as pd
import pronto


def read_hpo_terms(filepath):
    """
    Read the HPO ontology (.obo file) into a DataFrame of terms.
    Includes direct parent term IDs (useful later for ontology-based
    phenotype similarity scoring).
    """
    ontology = pronto.Ontology(filepath, encoding="ISO-8859-1")

    rows = []
    for term in ontology.terms():
        rows.append({
            "HPO_ID": term.id,
            "Name": term.name,
            "Definition": term.definition,
            "Parents": ";".join([parent.id for parent in term.superclasses(distance=1)])
        })

    return pd.DataFrame(rows)


def read_hpo_annotations(filepath):
    """
    Read the disease-to-phenotype annotation file (.hpoa).
    Loaded as all-string dtype since several columns (qualifier, evidence,
    sex, modifier) have inconsistent/mixed content across rows.
    No cleaning here — that happens in clean_hpo.py.
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        comment="#",
        dtype=str
    )
    return df


if __name__ == "__main__":
    terms_df = read_hpo_terms("data/raw/hpo/hp.obo")
    print(f"Read {len(terms_df)} HPO terms")
    terms_df.to_csv("data/interim/hpo_tables/hpo_terms.csv", index=False)

    annotations_df = read_hpo_annotations("data/raw/hpo/phenotype.hpoa")
    print(f"Read {len(annotations_df)} disease-phenotype annotation rows")
    annotations_df.to_csv(
        "data/interim/hpo_tables/disease_hpo_annotations_raw.csv",
        index=False
    )