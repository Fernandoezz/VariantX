from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import tempfile
import os

from pipeline import (
    parse_patient_vcf, parse_patient_cnv_vcf, load_reference_tables,
    annotate_variants, annotate_cnv_variants, score_patient_variants,
    predict_and_rank, predict_and_rank_cnv, explain_top_candidate,
    detect_vcf_type
)

app = FastAPI(title="VariantX API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend domain before going live
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {}


@app.on_event("startup")
def load_everything_once():
    """Load reference tables ONCE when the server starts, not per-request -
    same reasoning as Streamlit's @st.cache_resource, just done explicitly here."""
    print("Loading reference tables...")
    state["reference_tables"] = load_reference_tables()
    state["hpo_terms"] = pd.read_csv("../data/interim/hpo_tables/hpo_terms.csv")
    state["hpo_annotations"] = pd.read_csv(
        "../data/interim/hpo_tables/disease_hpo_annotations_clean.csv", low_memory=False
    )
    print("Reference tables loaded.")


def clean_for_json(df: pd.DataFrame):
    """Replace NaN/NaT with None so the DataFrame serializes to valid JSON.
    Categorical columns are cast to plain object dtype first, since
    .where() does not reliably replace missing values inside category
    dtype columns, silently leaving raw NaN in place otherwise."""
    df = df.astype(object)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/hpo/search")
def search_hpo(q: str):
    """Live symptom search - same logic as the Streamlit searchbox."""
    if not q or len(q) < 2:
        return []

    hpo_terms = state["hpo_terms"]
    matches = hpo_terms[hpo_terms["Name"].str.contains(q, case=False, na=False)].head(15)

    return [
        {"hpo_id": row["HPO_ID"], "name": row["Name"]}
        for _, row in matches.iterrows()
    ]


@app.post("/analyze")
async def analyze(
    mode: str = Form(...),           # "snv" or "cnv"
    symptoms: str = Form(...),       # JSON-encoded list of HPO IDs, e.g. '["HP:0001250"]'
    file: UploadFile = File(...)
):
    if mode not in ("snv", "cnv"):
        raise HTTPException(status_code=400, detail="mode must be 'snv' or 'cnv'")

    try:
        patient_hpo_terms = json.loads(symptoms)
        if isinstance(patient_hpo_terms, str):
            patient_hpo_terms = [patient_hpo_terms]
    except json.JSONDecodeError:
        # Fall back to plain comma-separated string, e.g. "HP:0003002,HP:0001250"
        patient_hpo_terms = [s.strip() for s in symptoms.split(",") if s.strip()]

    if not patient_hpo_terms:
        raise HTTPException(status_code=400, detail="At least one symptom (HPO ID) is required.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        detected_type = detect_vcf_type(tmp_path)

        reference_tables = state["reference_tables"]
        hpo_annotations = state["hpo_annotations"]

        if mode == "snv":
            patient_variants = parse_patient_vcf(tmp_path)
            if len(patient_variants) == 0:
                raise HTTPException(status_code=400, detail="No SNV/indel variants could be parsed from this VCF.")

            annotated = annotate_variants(patient_variants, reference_tables)
            scored = score_patient_variants(annotated, patient_hpo_terms, reference_tables["omim"], hpo_annotations)
            ranked, X, model = predict_and_rank(scored)

        else:
            patient_variants = parse_patient_cnv_vcf(tmp_path)
            if len(patient_variants) == 0:
                raise HTTPException(status_code=400, detail="No CNV deletions could be parsed from this VCF.")

            annotated = annotate_cnv_variants(patient_variants, reference_tables)
            scored = score_patient_variants(annotated, patient_hpo_terms, reference_tables["omim"], hpo_annotations)
            ranked, X, model = predict_and_rank_cnv(scored)

        supporting, opposing = explain_top_candidate(ranked, X, model)

        display_columns = (
            ["rank", "gene_symbol", "variant_id", "relevance_score",
             "phenotype_similarity_score", "best_matching_disease",
             "clinvar_significance", "in_clinvar"]
            if mode == "snv" else
            ["rank", "gene_symbol", "variant_id", "relevance_score",
             "phenotype_similarity_score", "best_matching_disease",
             "deletion_size", "overlap_fraction_of_gene"]
        )
        display_columns = [c for c in display_columns if c in ranked.columns]

        return {
            "detected_type": detected_type,
            "requested_mode": mode,
            "type_mismatch": detected_type != mode,
            "n_variants_parsed": len(patient_variants),
            "results": clean_for_json(ranked[display_columns]),
            "top_candidate_explanation": {
                "supporting": clean_for_json(supporting),
                "opposing": clean_for_json(opposing),
            }
        }

    finally:
        os.unlink(tmp_path)