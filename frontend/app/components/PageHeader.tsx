export default function PageHeader({
  icon,
  title,
  subtitle,
}: {
  icon: string;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="bg-gradient-to-b from-[#0a1628] to-[#0f1f38] px-6 py-14">
      <div className="max-w-3xl mx-auto">
        <div className="w-11 h-11 rounded-xl bg-blue-500/20 flex items-center justify-center mb-4">
          <i className={`ti ${icon} text-blue-400 text-xl`}></i>
        </div>
        <h1 className="text-3xl font-semibold text-white mb-2">{title}</h1>
        <p className="text-white/50">{subtitle}</p>
      </div>
    </div>
  );
}