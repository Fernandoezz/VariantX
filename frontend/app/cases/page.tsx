import PageHeader from "../components/PageHeader";

export default function MyCases() {
  return (
    <div>
      <PageHeader
        icon="ti-folder"
        title="My cases"
        subtitle="Your saved analysis history"
      />
      <div className="max-w-3xl mx-auto px-6 py-10">
        <div className="border border-dashed border-gray-300 rounded-2xl p-14 text-center">
          <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-4">
            <i className="ti ti-lock text-gray-400 text-xl"></i>
          </div>
          <p className="text-gray-500 text-sm mb-5">
            Sign in to save and revisit your analysis history.
          </p>
          <button className="bg-blue-500 text-white text-sm px-6 py-2.5 rounded-full font-medium hover:bg-blue-600 transition-colors">
            Sign in
          </button>
        </div>
      </div>
    </div>
  );
}