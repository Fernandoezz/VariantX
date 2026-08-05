import PageHeader from "../components/PageHeader";

export default function Releases() {
  return (
    <div>
      <PageHeader
        icon="ti-timeline"
        title="Release updates"
        subtitle="Version history and changelog"
      />
      <div className="max-w-3xl mx-auto px-6 py-10">
        <div className="border-l-2 border-blue-500 pl-5 py-1">
          <p className="text-sm font-semibold">v1.0 — Initial release</p>
          <p className="text-xs text-gray-500 mt-1.5 leading-relaxed">
            SNV and CNV prioritization models, phenotype-linked symptom search, and
            SHAP-based explainability.
          </p>
        </div>
      </div>
    </div>
  );
}