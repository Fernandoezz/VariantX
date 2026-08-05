import PageHeader from "../components/PageHeader";

export default function About() {
  return (
    <div>
      <PageHeader
        icon="ti-info-circle"
        title="About VariantX"
        subtitle="A patient-specific framework for genetic variant prioritization"
      />
      <div className="max-w-3xl mx-auto px-6 py-10 space-y-4 text-sm text-gray-600 leading-relaxed">
        <p>
          VariantX integrates phenotype similarity, gene-level constraint, structural
          dosage sensitivity, and inheritance consistency into a single, explainable
          ranking system for genetic variant prioritization in rare disease diagnosis.
        </p>
        <p>
          Unlike pathogenicity-only tools, VariantX considers each patient&apos;s
          individual clinical presentation. Unlike phenotype-only tools, it also
          incorporates gene-level and structural evidence, and provides SHAP-based
          explanations for every prediction.
        </p>
        <p>
          VariantX supports both single nucleotide variants and copy number variant
          deletions within a single, unified interface.
        </p>
      </div>
    </div>
  );
}