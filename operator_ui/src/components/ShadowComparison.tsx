"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ShadowData {
  shadow_tvd: number;
  action_distribution: Record<string, number>;
  shadow_action_deltas: Record<string, number>;
}

export default function ShadowComparison() {
  const [data, setData] = useState<ShadowData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/runs`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((runsData) => {
        const runs = runsData.runs ?? [];
        if (runs.length > 0) {
          const latest = runs[0];
          if (latest.action_distribution && latest.shadow_tvd !== undefined && latest.shadow_tvd !== null) {
            setData(null);
          }
        }
      })
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-slate-400">Loading shadow data...</div>;

  if (!data) {
    return (
      <div>
        <h2 className="text-lg font-semibold mb-1 text-gray-900 dark:text-white">Shadow Comparison</h2>
        <p className="text-xs text-slate-400 mb-3">
          Side-by-side comparison of production vs candidate rule distributions.
        </p>
        <p className="text-slate-400 text-sm">No shadow data available. Run automation with shadow_rules_config to see comparison.</p>
      </div>
    );
  }

  const deltaEntries = Object.entries(data.shadow_action_deltas);
  const maxDelta = Math.max(...deltaEntries.map(([, v]) => Math.abs(v)), 0.01);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1 text-gray-900 dark:text-white">
        Shadow Comparison
        <span className={`ml-2 text-sm px-2 py-0.5 rounded ${
          data.shadow_tvd > 0.15
            ? "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300"
            : "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
        }`}>
          TVD: {data.shadow_tvd.toFixed(3)}
        </span>
      </h2>
      <p className="text-xs text-slate-400 mb-3">
        Side-by-side comparison of production vs candidate rule distributions.
      </p>
      <div className="space-y-2">
        {deltaEntries.map(([action, delta]) => (
          <div key={action} className="flex items-center gap-3">
            <span className="text-xs w-40 truncate text-gray-700 dark:text-slate-300">{action}</span>
            <div className="flex-1 flex items-center">
              <div className="w-full bg-gray-100 dark:bg-slate-700 rounded h-4 relative">
                <div
                  className={`h-4 rounded absolute ${delta >= 0 ? "bg-blue-500 left-1/2" : "bg-orange-500 right-1/2"}`}
                  style={{ width: `${(Math.abs(delta) / maxDelta) * 50}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-px h-full bg-gray-300 dark:bg-slate-500" />
                </div>
              </div>
            </div>
            <span className={`text-xs font-mono w-16 text-right ${
              delta > 0 ? "text-blue-600 dark:text-blue-400" : "text-orange-600 dark:text-orange-400"
            }`}>
              {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
