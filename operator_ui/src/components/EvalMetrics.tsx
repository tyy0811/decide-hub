"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function EvalMetrics() {
  const [policy, setPolicy] = useState("popularity");
  const [k, setK] = useState(10);
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchMetrics = () => {
    setLoading(true);
    fetch(`${API_BASE}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ policy, k }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => setMetrics(data.metrics ?? null))
      .catch(() => setMetrics(null))
      .finally(() => setLoading(false));
  };

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1 text-gray-900 dark:text-white">Evaluation Metrics</h2>
      <p className="text-xs text-slate-400 mb-3">Run offline evaluation on MovieLens 1M test set. Measures NDCG, MRR, and HitRate at K.</p>
      <div className="flex gap-2 mb-3">
        <select
          value={policy}
          onChange={(e) => setPolicy(e.target.value)}
          className="px-2 py-1 rounded border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-gray-900 dark:text-white text-sm"
        >
          <option value="popularity">popularity</option>
          <option value="scorer">scorer</option>
        </select>
        <label className="flex items-center gap-1 text-sm text-gray-600 dark:text-slate-400">
          K=
          <input
            type="number"
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="w-14 px-2 py-1 rounded border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-gray-900 dark:text-white text-sm"
          />
        </label>
        <button
          onClick={fetchMetrics}
          disabled={loading}
          className="px-3 py-1 rounded bg-blue-600 text-white text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "..." : "Evaluate"}
        </button>
      </div>
      {metrics ? (
        <div className="space-y-2">
          {Object.entries(metrics).map(([name, value]) => (
            <div key={name} className="flex justify-between items-center">
              <span className="text-sm text-gray-700 dark:text-slate-300 font-mono">{name}</span>
              <span className="text-sm font-bold text-gray-900 dark:text-white">{value.toFixed(4)}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-slate-400 text-sm">
          {loading ? "Running evaluation on MovieLens 1M..." : "Click Evaluate to run offline metrics"}
        </p>
      )}
    </div>
  );
}
