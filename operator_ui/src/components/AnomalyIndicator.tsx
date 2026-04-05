"use client";

import { useEffect, useState } from "react";
import { authFetch } from "@/lib/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface AnomalyData {
  status: string;
  anomalies: { metric: string; observed: number; expected_range: string; severity: string }[];
  baseline_window: number;
  recent_window: number;
}

export default function AnomalyIndicator() {
  const [data, setData] = useState<AnomalyData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    authFetch(`${API_BASE}/anomalies`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-slate-400 text-sm">Checking anomalies...</div>;

  if (!data) {
    return (
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 rounded-full bg-gray-400" />
        <span className="text-sm text-slate-400">Anomaly check unavailable</span>
      </div>
    );
  }

  const color = data.status === "ok"
    ? "bg-green-500"
    : "bg-red-500";
  const label = data.status === "ok"
    ? "No anomalies detected"
    : `${data.anomalies.length} anomal${data.anomalies.length === 1 ? "y" : "ies"} detected`;

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-3 h-3 rounded-full ${color}`} />
        <span className="text-sm font-medium text-gray-900 dark:text-white">{label}</span>
        <span className="text-xs text-slate-400">
          ({data.recent_window} recent vs {data.baseline_window} baseline runs)
        </span>
      </div>
      {data.anomalies.length > 0 && (
        <div className="space-y-1">
          {data.anomalies.map((a, i) => (
            <div key={i} className="text-xs text-red-600 dark:text-red-400">
              <span className="font-mono">{a.metric}</span>: observed {a.observed}, expected {a.expected_range}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
