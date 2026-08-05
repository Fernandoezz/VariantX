import PageHeader from "../components/PageHeader";

const topics = [
  {
    icon: "ti-chart-bar",
    title: "How ranking works",
    text: "VariantX combines clinical significance, gene-level constraint, structural dosage sensitivity, inheritance consistency, and patient-specific phenotype similarity into a single relevance score for each candidate variant.",
  },
  {
    icon: "ti-dna",
    title: "Phenotype matching",
    text: "Symptoms are compared against each candidate gene's known disease associations using the Human Phenotype Ontology, weighted by how characteristic each symptom is of that disease.",
  },
  {
    icon: "ti-bulb",
    title: "Explainability",
    text: "Every ranked result includes supporting and opposing evidence, generated using SHAP, showing exactly which factors influenced the model's prediction.",
  },
];

export default function KnowledgeBase() {
  return (
    <div>
      <PageHeader
        icon="ti-book"
        title="Knowledge base"
        subtitle="How VariantX evaluates and ranks candidate variants"
      />
      <div className="max-w-3xl mx-auto px-6 py-10 space-y-4">
        {topics.map((t) => (
          <div
            key={t.title}
            className="border border-gray-200 rounded-2xl p-6 hover:border-blue-300 hover:shadow-sm transition-all"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-9 h-9 rounded-lg bg-blue-50 flex items-center justify-center">
                <i className={`ti ${t.icon} text-blue-600`}></i>
              </div>
              <h2 className="font-semibold">{t.title}</h2>
            </div>
            <p className="text-sm text-gray-600 leading-relaxed">{t.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}