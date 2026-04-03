"use client";

import { useEffect, useState } from "react";

interface Run {
  run_id: string;
  status: string;
  entities_processed: number;
  entities_failed: number;
  action_distribution: Record<string, number>;
  started_at: string;
  completed_at: string | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function RunsTable() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/runs`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => setRuns(data.runs ?? []))
      .catch(() => setRuns([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-slate-400">Loading runs...</div>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">Recent Automation Runs</h2>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-200 dark:border-slate-600">
            <th className="text-left py-2 pr-4 text-gray-700 dark:text-slate-300">Run ID</th>
            <th className="text-left py-2 pr-4 text-gray-700 dark:text-slate-300">Status</th>
            <th className="text-right py-2 pr-4 text-gray-700 dark:text-slate-300">Processed</th>
            <th className="text-right py-2 pr-4 text-gray-700 dark:text-slate-300">Failed</th>
            <th className="text-left py-2 text-gray-700 dark:text-slate-300">Started</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id} className="border-b border-gray-100 dark:border-slate-700">
              <td className="py-2 pr-4 font-mono text-xs text-gray-800 dark:text-slate-200">{run.run_id}</td>
              <td className="py-2 pr-4">
                <span
                  className={`px-2 py-0.5 rounded text-xs font-medium ${
                    run.status === "completed"
                      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
                      : "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300"
                  }`}
                >
                  {run.status}
                </span>
              </td>
              <td className="py-2 pr-4 text-right text-gray-800 dark:text-slate-200">{run.entities_processed}</td>
              <td className="py-2 pr-4 text-right text-gray-800 dark:text-slate-200">{run.entities_failed}</td>
              <td className="py-2 text-xs text-gray-500 dark:text-slate-400">
                {new Date(run.started_at).toLocaleString()}
              </td>
            </tr>
          ))}
          {runs.length === 0 && (
            <tr>
              <td colSpan={5} className="py-4 text-center text-slate-400">
                No runs yet
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
