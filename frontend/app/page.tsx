"use client";

import { useState } from "react";

type HpoTerm = {
  hpo_id: string;
  name: string;
};

type RankedVariant = {
  rank: number;
  gene_symbol: string;
  variant_id: string;
  relevance_score: number;
  phenotype_similarity_score: number;
  best_matching_disease: string | null;
  [key: string]: unknown;
};

type AnalyzeResponse = {
  detected_type: string;
  requested_mode: string;
  type_mismatch: boolean;
  n_variants_parsed: number;
  results: RankedVariant[];
  top_candidate_explanation: {
    supporting: { feature: string; value: unknown; shap_value: number }[];
    opposing: { feature: string; value: unknown; shap_value: number }[];
  };
};

const API_BASE = "http://localhost:8000";

export default function Home() {
  const [mode, setMode] = useState<"snv" | "cnv">("snv");
  const [file, setFile] = useState<File | null>(null);
  const [symptomQuery, setSymptomQuery] = useState("");
  const [suggestions, setSuggestions] = useState<HpoTerm[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<HpoTerm[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSymptomSearch(query: string) {
    setSymptomQuery(query);
    if (query.length < 2) {
      setSuggestions([]);
      return;
    }
    const res = await fetch(
      `${API_BASE}/hpo/search?q=${encodeURIComponent(query)}`,
    );
    const data = await res.json();
    setSuggestions(data);
  }

  function addSymptom(term: HpoTerm) {
    if (!selectedSymptoms.find((s) => s.hpo_id === term.hpo_id)) {
      setSelectedSymptoms([...selectedSymptoms, term]);
    }
    setSymptomQuery("");
    setSuggestions([]);
  }

  function removeSymptom(hpoId: string) {
    setSelectedSymptoms(selectedSymptoms.filter((s) => s.hpo_id !== hpoId));
  }

  async function runAnalysis() {
    if (!file || selectedSymptoms.length === 0) {
      setError("Please upload a VCF file and add at least one symptom.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("mode", mode);
    formData.append(
      "symptoms",
      JSON.stringify(selectedSymptoms.map((s) => s.hpo_id)),
    );
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Analysis failed.");
      }

      const data: AnalyzeResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      {/* Dark hero section with depth */}
      <div className="relative bg-[#0a1628] pt-20 pb-28 px-6 overflow-hidden">
        {/* Background blobs */}
        <div className="absolute -top-24 -left-24 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl"></div>
        <div className="absolute top-1/3 -right-32 w-[28rem] h-[28rem] bg-purple-600/15 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-1/4 w-72 h-72 bg-teal-500/10 rounded-full blur-3xl"></div>

        <div className="relative max-w-3xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-semibold text-white mb-3 tracking-tight">
            The future of variant prioritization
          </h1>
          <p className="text-white/50 mb-10">
            Patient-specific, explainable ranking for SNVs and CNV deletions
          </p>

          <div className="flex justify-center gap-2 mb-6">
            <button
              onClick={() => setMode("snv")}
              className={`px-5 py-2 rounded-full text-xs font-medium tracking-wide uppercase transition-colors ${
                mode === "snv"
                  ? "bg-blue-500 text-white"
                  : "bg-white/5 text-white/50 hover:bg-white/10"
              }`}
            >
              SNV / Indel
            </button>
            <button
              onClick={() => setMode("cnv")}
              className={`px-5 py-2 rounded-full text-xs font-medium tracking-wide uppercase transition-colors ${
                mode === "cnv"
                  ? "bg-blue-500 text-white"
                  : "bg-white/5 text-white/50 hover:bg-white/10"
              }`}
            >
              CNV Deletion
            </button>
          </div>

          <div className="bg-white/95 backdrop-blur rounded-2xl p-6 text-left shadow-2xl shadow-black/40 border border-white/10">
            <div className="flex items-center gap-2 mb-5">
              <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center">
                <i className="ti ti-file-upload text-blue-600 text-sm"></i>
              </div>
              <label className="text-sm font-medium text-gray-700">Patient VCF</label>
            </div>
            <input
              type="file"
              accept=".vcf,.gz"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="text-sm mb-5 ml-10 text-gray-600"
            />

            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center">
                <i className="ti ti-stethoscope text-blue-600 text-sm"></i>
              </div>
              <label className="text-sm font-medium text-gray-700">Clinical symptoms</label>
            </div>
            <div className="relative ml-10">
              <i className="ti ti-search absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 text-sm"></i>
              <input
                type="text"
                value={symptomQuery}
                onChange={(e) => handleSymptomSearch(e.target.value)}
                placeholder="Search symptoms, e.g. seizure"
                className="w-full border border-gray-200 rounded-lg pl-9 pr-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-400"
              />

              {suggestions.length > 0 && (
                <div className="absolute w-full bg-white border border-gray-200 rounded-lg mt-1 max-h-48 overflow-y-auto shadow-lg z-10">
                  {suggestions.map((term) => (
                    <div
                      key={term.hpo_id}
                      onClick={() => addSymptom(term)}
                      className="px-3 py-2 text-sm hover:bg-gray-50 cursor-pointer"
                    >
                      {term.name} <span className="text-gray-400">({term.hpo_id})</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="flex flex-wrap gap-2 mt-3">
                {selectedSymptoms.map((s) => (
                  <span
                    key={s.hpo_id}
                    className="text-xs bg-blue-50 text-blue-700 pl-3 pr-2 py-1 rounded-full flex items-center gap-1.5"
                  >
                    {s.name}
                    <button
                      onClick={() => removeSymptom(s.hpo_id)}
                      className="hover:bg-blue-100 rounded-full w-4 h-4 flex items-center justify-center"
                    >
                      <i className="ti ti-x text-[10px]"></i>
                    </button>
                  </span>
                ))}
              </div>
            </div>

            <button
              onClick={runAnalysis}
              disabled={loading}
              className="w-full bg-blue-500 text-white py-3 rounded-lg text-sm font-medium mt-6 disabled:opacity-50 hover:bg-blue-600 transition-colors flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <i className="ti ti-loader-2 animate-spin"></i> Running analysis...
                </>
              ) : (
                <>
                  <i className="ti ti-player-play-filled text-xs"></i> Run analysis
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results section, white background */}
      <div className="max-w-3xl mx-auto px-6 py-10">
        {error && (
          <p className="text-red-600 text-sm mb-4 flex items-center gap-2">
            <i className="ti ti-alert-circle"></i> {error}
          </p>
        )}

        {result && (
          <div>
            <h2 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
              <i className="ti ti-list-details text-gray-400"></i>
              Ranked candidates ({result.n_variants_parsed} variants parsed)
            </h2>
            {result.results.map((variant) => (
              <div
                key={`${variant.variant_id}_${variant.gene_symbol}_${variant.rank}`}
                className={`border rounded-xl p-4 mb-2 transition-shadow hover:shadow-sm ${
                  variant.rank === 1
                    ? "border-blue-500 border-2 bg-blue-50/30"
                    : "border-gray-200"
                }`}
              >
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-3">
                    <span
                      className={`text-xs px-2 py-1 rounded-full font-medium ${
                        variant.rank === 1
                          ? "bg-blue-500 text-white"
                          : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      Rank {variant.rank}
                    </span>
                    <span className="font-medium">{variant.gene_symbol}</span>
                    <span className="text-xs text-gray-500">{variant.variant_id}</span>
                  </div>
                  <span className="font-medium text-blue-600">
                    {variant.relevance_score.toFixed(4)}
                  </span>
                </div>
                {variant.best_matching_disease && (
                  <p className="text-xs text-gray-500 mt-2 flex items-center gap-1.5">
                    <i className="ti ti-stethoscope text-gray-400"></i>
                    Best match: {variant.best_matching_disease}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}