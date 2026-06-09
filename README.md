# VariantX

## Overview

VariantX is a machine learning framework for genomic variant prioritization and pathogenicity prediction. The project aims to assist in identifying disease-causing variants by integrating information from multiple genomic databases and applying explainable machine learning techniques.

This project is being developed as a Final Year Project (FYP).

---

## Objectives

* Develop a pathogenicity prediction model for Single Nucleotide Variants (SNVs).
* Develop a pathogenicity prediction model for Copy Number Variant (CNV) deletions.
* Integrate information from multiple genomic databases.
* Provide interpretable predictions using explainable AI techniques.
* Improve variant prioritization in rare disease analysis.

---

## Models

### SNV Model

Predicts the pathogenicity of single nucleotide variants using variant-level, gene-level, and phenotype-based features.

### CNV Deletion Model

Predicts the pathogenicity of copy number deletions using genomic, dosage sensitivity, and phenotype similarity features.

---

## Databases Used

* ClinVar
* Human Phenotype Ontology (HPO)
* OMIM
* gnomAD
* dbNSFP

---

## Technologies

### Programming Language

* Python

### Machine Learning

* LightGBM
* Scikit-learn

### Explainability

* SHAP

### Data Processing

* Pandas
* NumPy
* Jupyter Notebook

---

## Project Structure

```text
VariantX/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── final/
│
├── logs/
│
├── models/
│   ├── snv_model/
│   └── cnv_model/
│
├── notebooks/
│   ├── 01_clinvar_exploration.ipynb
│   ├── 02_hpo_exploration.ipynb
│   ├── 03_gnomad_exploration.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_training.ipynb
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   ├── shap_plots/
│   └── tables/
│
├── scripts/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── training/
│   └── explainability/
│
├── requirements.txt
└── README.md
```

---

## Current Progress

### Completed

* ClinVar preprocessing
* HPO preprocessing
* gnomAD preprocessing

### In Progress

* dbNSFP integration
* OMIM integration
* Feature engineering

### Upcoming

* SNV model training
* CNV deletion model development
* SHAP explainability analysis

---

## Explainability

Model predictions will be interpreted using SHAP (SHapley Additive exPlanations) to identify the most influential features contributing to variant pathogenicity.

---

## Project Status

🚧 Under Development

---

## Author

Fernando Fernandez

Final Year Project

Department of Computer Science and Engineering

University of Moratuwa
