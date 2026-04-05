"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface RunDetail {
  run_id: string;
  status: string;
  entities_processed: number;
  entities_failed: number;
  action_distribution: Record<string, number>;
  started_at: string;
  completed_at: string | null;
  outcomes: { entity_id: string; action_taken: string; rule_matched: string | null; permission_result: string }[];
  audit_events: { entity_id: string; actor: string; action_type: string; reason: string | null }[];
}

export default function RunDetailPage() {
  const params = useParams<{ id: string }>();
  const runId = params.id;
  const [data, setData] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/runs/${runId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return <main className="max-w-7xl mx-auto p-8"><p className="text-slate-400">Loading run...</p></main>;
  if (error) return <main className="max-w-7xl mx-auto p-8"><p className="text-red-500">Error: {error}</p></main>;
  if (!data) return <main className="max-w-7xl mx-auto p-8"><p className="text-slate-400">Run not found</p></main>;

  return (
    <main className="max-w-7xl mx-auto p-8">
      <div className="mb-4">
        <Link href="/" className="text-blue-600 dark:text-blue-400 text-sm hover:underline">&larr; Back to Overview</Link>
      </div>
      <h1 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">Run {data.run_id}</h1>
      <div className="flex gap-4 text-sm text-slate-500 mb-6">
        <span>Status: <span className="font-medium text-gray-900 dark:text-white">{data.status}</span></span>
        <span>Processed: {data.entities_processed}</span>
        <span>Failed: {data.entities_failed}</span>
        <span>Started: {data.started_at}</span>
      </div>

      <h2 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">Entity Outcomes</h2>
      <div className="overflow-x-auto mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-slate-700">
              <th className="text-left py-2 px-3 text-gray-600 dark:text-slate-400">Entity</th>
              <th className="text-left py-2 px-3 text-gray-600 dark:text-slate-400">Action</th>
              <th className="text-left py-2 px-3 text-gray-600 dark:text-slate-400">Rule</th>
              <th className="text-left py-2 px-3 text-gray-600 dark:text-slate-400">Permission</th>
            </tr>
          </thead>
          <tbody>
            {data.outcomes.map((o, i) => (
              <tr key={i} className="border-b border-gray-100 dark:border-slate-800">
                <td className="py-2 px-3 font-mono text-gray-700 dark:text-slate-300">{o.entity_id}</td>
                <td className="py-2 px-3 text-gray-900 dark:text-white">{o.action_taken}</td>
                <td className="py-2 px-3 text-slate-500">{o.rule_matched || "\u2014"}</td>
                <td className="py-2 px-3">
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    o.permission_result === "allowed" ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300" :
                    o.permission_result === "blocked" ? "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300" :
                    "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300"
                  }`}>{o.permission_result}</span>
                </td>
              </tr>
            ))}
            {data.outcomes.length === 0 && (
              <tr><td colSpan={4} className="py-4 text-center text-slate-400">No outcomes recorded</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {data.audit_events.length > 0 && (
        <>
          <h2 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">Audit Events</h2>
          <div className="space-y-1">
            {data.audit_events.map((a, i) => (
              <div key={i} className="text-sm text-slate-500">
                <span className="font-mono text-gray-700 dark:text-slate-300">{a.entity_id}</span>
                {" \u2014 "}
                <span className="text-gray-900 dark:text-white">{a.action_type}</span>
                {" by "}
                <span>{a.actor}</span>
                {a.reason && <span className="text-slate-400"> ({a.reason})</span>}
              </div>
            ))}
          </div>
        </>
      )}
    </main>
  );
}
