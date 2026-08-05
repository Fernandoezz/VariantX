import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "VariantX",
  description: "Patient-specific variant prioritization",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/iconfont/tabler-icons.min.css"
        />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-white text-gray-900" style={{ fontFamily: "Inter, sans-serif" }}>
        <nav className="bg-[#0a1628] border-b border-white/10 sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 text-lg font-semibold text-white">
              <div className="w-7 h-7 rounded-lg bg-blue-500 flex items-center justify-center">
                <i className="ti ti-dna text-sm"></i>
              </div>
              Variant<span className="text-blue-400">X</span>
            </Link>

            <div className="flex items-center gap-7 text-xs font-medium tracking-wide uppercase text-white/60">
              <Link href="/" className="text-white border-b-2 border-blue-400 pb-1">
                Search
              </Link>
              <Link href="/knowledge-base" className="hover:text-white transition-colors">
                Knowledge Base
              </Link>
              <Link href="/cases" className="hover:text-white transition-colors">
                My Cases
              </Link>
              <div className="relative group">
                <button className="hover:text-white transition-colors flex items-center gap-1 normal-case">
                  Learn more <i className="ti ti-chevron-down text-xs"></i>
                </button>
                <div className="absolute right-0 top-full pt-3 hidden group-hover:block">
                  <div className="bg-white border border-gray-200 rounded-xl shadow-xl py-2 w-48 normal-case">
                    <Link href="/about" className="flex items-center gap-2 px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50">
                      <i className="ti ti-info-circle text-gray-400"></i> About VariantX
                    </Link>
                    <Link href="/releases" className="flex items-center gap-2 px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50">
                      <i className="ti ti-timeline text-gray-400"></i> Release updates
                    </Link>
                  </div>
                </div>
              </div>
              <button className="hover:text-white transition-colors normal-case">Log in</button>
              <button className="bg-blue-500 text-white px-4 py-2 rounded-full hover:bg-blue-600 transition-colors font-medium normal-case">
                Sign up
              </button>
            </div>
          </div>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}