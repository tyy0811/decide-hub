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

  if (loading) return <div className="text-gray-500">Loading runs...</div>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-3">Recent Automation Runs</h2>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-2 pr-4">Run ID</th>
            <th className="text-left py-2 pr-4">Status</th>
            <th className="text-right py-2 pr-4">Processed</th>
            <th className="text-right py-2 pr-4">Failed</th>
            <th className="text-left py-2">Started</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id} className="border-b border-gray-100">
              <td className="py-2 pr-4 font-mono text-xs">{run.run_id}</td>
              <td className="py-2 pr-4">
                <span
                  className={`px-2 py-0.5 rounded text-xs ${
                    run.status === "completed"
                      ? "bg-green-100 text-green-800"
                      : "bg-yellow-100 text-yellow-800"
                  }`}
                >
                  {run.status}
                </span>
              </td>
              <td className="py-2 pr-4 text-right">{run.entities_processed}</td>
              <td className="py-2 pr-4 text-right">{run.entities_failed}</td>
              <td className="py-2 text-xs text-gray-500">
                {new Date(run.started_at).toLocaleString()}
              </td>
            </tr>
          ))}
          {runs.length === 0 && (
            <tr>
              <td colSpan={5} className="py-4 text-center text-gray-400">
                No runs yet
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
