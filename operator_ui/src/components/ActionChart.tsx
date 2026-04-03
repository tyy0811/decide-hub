"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const ACTION_COLORS: Record<string, string> = {
  priority_outreach: "bg-green-500",
  standard_sequence: "bg-blue-500",
  flag_for_review: "bg-yellow-500",
  deprioritize: "bg-gray-400",
  "send_external_email:approval_required": "bg-orange-500",
  "delete_lead:blocked": "bg-red-500",
};

export default function ActionChart() {
  const [distribution, setDistribution] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/runs`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => {
        const runs = data.runs ?? [];
        if (runs.length > 0) {
          setDistribution(runs[0].action_distribution || {});
        }
      })
      .catch(() => setDistribution({}))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-slate-400">Loading chart...</div>;

  const entries = Object.entries(distribution);
  const maxVal = Math.max(...entries.map(([, v]) => v), 1);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1 text-gray-900 dark:text-white">Action Distribution</h2>
      <p className="text-xs text-slate-400 mb-3">Breakdown of routing decisions from the most recent automation run.</p>
      {entries.length === 0 ? (
        <p className="text-slate-400 text-sm">No data yet</p>
      ) : (
        <div className="space-y-3">
          {entries.map(([action, count]) => (
            <div key={action} className="flex items-center gap-3">
              <span className="text-xs w-48 truncate text-gray-700 dark:text-slate-300">{action}</span>
              <div className="flex-1 bg-gray-100 dark:bg-slate-700 rounded h-6">
                <div
                  className={`h-6 rounded ${
                    ACTION_COLORS[action] || "bg-indigo-500"
                  }`}
                  style={{ width: `${(count / maxVal) * 100}%` }}
                />
              </div>
              <span className="text-xs font-mono w-8 text-right text-gray-800 dark:text-slate-200">{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
