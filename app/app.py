import streamlit as st
import pandas as pd
import sys
import os
from streamlit_searchbox import st_searchbox

sys.path.append(os.path.dirname(__file__))
from pipeline import (
    parse_patient_vcf, parse_patient_cnv_vcf, load_reference_tables,
    annotate_variants, annotate_cnv_variants, score_patient_variants,
    predict_and_rank, predict_and_rank_cnv, explain_top_candidate,
    detect_vcf_type
)


st.set_page_config(page_title="VariantX - Variant Prioritization", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #4F8BF9, #38C6D9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2 {
        font-size: 1.4rem !important;
        border-bottom: 2px solid #4F8BF9;
        padding-bottom: 0.4rem;
        margin-top: 2rem !important;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #4F8BF9;
        border-radius: 12px;
        padding: 1rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4F8BF9, #38C6D9);
        border: none;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.title("VariantX: Patient-Specific Variant Prioritization")
st.caption("Upload a patient VCF and enter clinical symptoms to rank candidate causal variants.")


if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []


def reset_app():
    st.session_state.selected_symptoms = []
    st.session_state.reset_counter += 1


col_reset, _ = st.columns([1, 5])
with col_reset:
    if st.button("🔄 New Analysis (Reset)"):
        reset_app()
        st.rerun()


@st.cache_resource
def get_reference_tables():
    return load_reference_tables()


@st.cache_data
def get_hpo_terms():
    return pd.read_csv("data/interim/hpo_tables/hpo_terms.csv")


@st.cache_data
def get_hpo_annotations():
    return pd.read_csv("data/interim/hpo_tables/disease_hpo_annotations_clean.csv", low_memory=False)


reference_tables = get_reference_tables()
hpo_terms_df = get_hpo_terms()
hpo_annotations_df = get_hpo_annotations()


st.header("0. Select Analysis Mode")
analysis_mode = st.radio(
    "What type of variants does this VCF contain?",
    ["SNV / Indel", "CNV Deletion"],
    horizontal=True
)


st.header("1. Upload Patient VCF")
uploaded_vcf = st.file_uploader(
    "Upload a VCF file (.vcf or .vcf.gz)",
    type=["vcf", "gz"],
    key=f"vcf_uploader_{st.session_state.reset_counter}"
)


st.header("2. Enter Patient Symptoms")
st.caption("Start typing a symptom name - matching HPO terms appear instantly.")


def search_hpo_terms(search_term: str):
    if not search_term or len(search_term) < 2:
        return []
    matches = hpo_terms_df[
        hpo_terms_df["Name"].str.contains(search_term, case=False, na=False)
    ].head(15)
    return [
        (f"{row['Name']} ({row['HPO_ID']})", row["HPO_ID"])
        for _, row in matches.iterrows()
    ]


selected_hpo_id = st_searchbox(
    search_hpo_terms,
    placeholder="Type a symptom name (e.g. 'seizure', 'developmental delay')...",
    key=f"symptom_searchbox_{st.session_state.reset_counter}",
)

if selected_hpo_id:
    already_added = any(s["hpo_id"] == selected_hpo_id for s in st.session_state.selected_symptoms)
    if not already_added:
        term_name = hpo_terms_df.loc[hpo_terms_df["HPO_ID"] == selected_hpo_id, "Name"].values[0]
        st.session_state.selected_symptoms.append({"hpo_id": selected_hpo_id, "name": term_name})
        st.rerun()

st.write("**Selected symptoms:**")
if st.session_state.selected_symptoms:
    for i, symptom in enumerate(st.session_state.selected_symptoms):
        col_a, col_b = st.columns([5, 1])
        col_a.write(f"- {symptom['name']} ({symptom['hpo_id']})")
        if col_b.button("Remove", key=f"remove_{i}"):
            st.session_state.selected_symptoms.pop(i)
            st.rerun()
else:
    st.write("_No symptoms added yet._")


st.header("3. Run Analysis")

if st.button("Run Analysis", type="primary"):
    if uploaded_vcf is None:
        st.error("Please upload a patient VCF file.")
    elif len(st.session_state.selected_symptoms) == 0:
        st.error("Please add at least one symptom.")
    else:
        temp_path = f"data/sample_patient/_uploaded_{uploaded_vcf.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_vcf.getbuffer())

        detected_type = detect_vcf_type(temp_path)
        expected_type = "cnv" if analysis_mode == "CNV Deletion" else "snv"

        if detected_type != expected_type:
            st.warning(
                f"⚠️ This file looks like it contains **{detected_type.upper()}**-style records, "
                f"but you selected **{analysis_mode}** mode. Results may be inaccurate - "
                f"consider switching modes."
            )

        patient_hpo_terms = [s["hpo_id"] for s in st.session_state.selected_symptoms]

        ranked = None
        X = None
        model = None
        display_columns = []

        try:
            if analysis_mode == "SNV / Indel":
                with st.spinner("Parsing VCF..."):
                    patient_variants = parse_patient_vcf(temp_path)

                if len(patient_variants) == 0:
                    st.error("No SNV/indel variants could be parsed from this VCF.")
                else:
                    st.success(f"Parsed {len(patient_variants)} variants.")

                    with st.spinner("Annotating variants against reference databases..."):
                        annotated = annotate_variants(patient_variants, reference_tables)

                    with st.spinner("Scoring phenotype similarity..."):
                        scored = score_patient_variants(
                            annotated, patient_hpo_terms, reference_tables["omim"], hpo_annotations_df
                        )

                    with st.spinner("Running SNV prioritization model..."):
                        ranked, X, model = predict_and_rank(scored)

                    display_columns = [
                        "rank", "gene_symbol", "variant_id", "relevance_score",
                        "phenotype_similarity_score", "best_matching_disease",
                        "clinvar_significance", "in_clinvar"
                    ]

            else:  # CNV Deletion
                with st.spinner("Parsing CNV VCF..."):
                    patient_variants = parse_patient_cnv_vcf(temp_path)

                if len(patient_variants) == 0:
                    st.error("No CNV deletions (SVTYPE=DEL with END field) could be parsed from this VCF.")
                else:
                    st.success(f"Parsed {len(patient_variants)} deletions.")

                    with st.spinner("Annotating deletions against gene coordinates..."):
                        annotated = annotate_cnv_variants(patient_variants, reference_tables)

                    with st.spinner("Scoring phenotype similarity..."):
                        scored = score_patient_variants(
                            annotated, patient_hpo_terms, reference_tables["omim"], hpo_annotations_df
                        )

                    with st.spinner("Running CNV prioritization model..."):
                        ranked, X, model = predict_and_rank_cnv(scored)

                    display_columns = [
                        "rank", "gene_symbol", "variant_id", "relevance_score",
                        "phenotype_similarity_score", "best_matching_disease",
                        "deletion_size", "overlap_fraction_of_gene"
                    ]

        except ValueError as e:
            st.error(f"⚠️ Could not process this file: {e}")

        if ranked is not None:
            st.header("Results: Ranked Candidate Variants")
            st.dataframe(ranked[display_columns], use_container_width=True)

            csv_data = ranked[display_columns].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="variantx_results.csv",
                mime="text/csv"
            )

            st.subheader("Top Candidate Detail")
            top_variant = ranked.iloc[0]
            st.write(f"**Gene:** {top_variant['gene_symbol']}")
            st.write(f"**Variant:** {top_variant['variant_id']}")
            st.write(f"**Relevance score:** {top_variant['relevance_score']:.4f}")
            st.write(f"**Best matching disease:** {top_variant['best_matching_disease']}")

            if analysis_mode == "SNV / Indel" and not top_variant.get("in_clinvar", True):
                st.warning(
                    "This variant was not found in ClinVar. Ranking relies primarily on "
                    "gene-level and phenotype evidence rather than known clinical significance."
                )

            with st.spinner("Computing explanation for top candidate..."):
                try:
                    supporting, opposing = explain_top_candidate(ranked, X, model)

                    st.subheader("Why was this variant ranked #1?")
                    col_sup, col_opp = st.columns(2)

                    with col_sup:
                        st.markdown("**✅ Supporting evidence**")
                        for _, row in supporting.iterrows():
                            st.write(f"- `{row['feature']}` = {row['value']} (impact: +{row['shap_value']:.3f})")

                    with col_opp:
                        st.markdown("**⚠️ Opposing evidence**")
                        if len(opposing) > 0:
                            for _, row in opposing.iterrows():
                                st.write(f"- `{row['feature']}` = {row['value']} (impact: {row['shap_value']:.3f})")
                        else:
                            st.write("_No significant opposing evidence found._")
                except Exception as e:
                    st.info(f"Explanation could not be computed: {e}")