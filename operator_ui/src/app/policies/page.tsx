"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface EvalResult {
  policy: string;
  k: number;
  metrics: Record<string, number>;
}

export default function PoliciesPage() {
  const [results, setResults] = useState<EvalResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/evaluate/results`)
      .then((r) => r.json())
      .then((data) => setResults(data.results ?? []))
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="max-w-7xl mx-auto p-8">
      <h1 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">Policy Comparison</h1>

      {loading && <p className="text-slate-400">Loading evaluation results...</p>}

      {!loading && results.length === 0 && (
        <p className="text-slate-400">
          No cached evaluation results. Run <code className="bg-gray-100 dark:bg-slate-700 px-1 rounded">make eval</code> or
          call <code className="bg-gray-100 dark:bg-slate-700 px-1 rounded">POST /evaluate</code> to populate.
        </p>
      )}

      {results.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-slate-700">
                <th className="text-left py-2 px-3 text-gray-600 dark:text-slate-400">Policy</th>
                <th className="text-left py-2 px-3 text-gray-600 dark:text-slate-400">K</th>
                {results[0] && Object.keys(results[0].metrics).map((m) => (
                  <th key={m} className="text-left py-2 px-3 text-gray-600 dark:text-slate-400">{m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i} className="border-b border-gray-100 dark:border-slate-800">
                  <td className="py-2 px-3 font-medium text-gray-900 dark:text-white">{r.policy}</td>
                  <td className="py-2 px-3 text-slate-500">{r.k}</td>
                  {Object.values(r.metrics).map((v, j) => (
                    <td key={j} className="py-2 px-3 font-mono text-gray-700 dark:text-slate-300">{v.toFixed(4)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
