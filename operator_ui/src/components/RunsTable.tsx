"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import Link from "next/link";

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
  const [liveRun, setLiveRun] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const bufferRef = useRef<any[]>([]);

  const fetchRuns = useCallback(() => {
    fetch(`${API_BASE}/runs`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => setRuns(data.runs ?? []))
      .catch(() => setRuns([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  useEffect(() => {
    const WS_URL = API_BASE.replace("http", "ws") + "/ws/runs";
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        bufferRef.current.push(data);
      };

      ws.onerror = () => ws.close();
      ws.onclose = () => { wsRef.current = null; };
    } catch {
      // WebSocket unavailable — fall back to fetch-only
    }

    // Flush buffer every 500ms
    const interval = setInterval(() => {
      if (bufferRef.current.length === 0) return;
      const events = [...bufferRef.current];
      bufferRef.current = [];

      for (const evt of events) {
        if (evt.event === "run_started") {
          setLiveRun(evt.run_id);
        }
        if (evt.event === "run_completed") {
          setLiveRun(null);
          fetchRuns();
        }
      }
    }, 500);

    return () => {
      clearInterval(interval);
      wsRef.current?.close();
    };
  }, [fetchRuns]);

  if (loading) return <div className="text-slate-400">Loading runs...</div>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1 text-gray-900 dark:text-white">Recent Automation Runs</h2>
      <p className="text-xs text-slate-400 mb-3">History of automation pipeline executions showing entities processed, failures, and timing.</p>
      {liveRun && (
        <div className="text-xs text-blue-600 dark:text-blue-400 mb-2">
          Run {liveRun} in progress...
        </div>
      )}
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
              <td className="py-2 pr-4 font-mono text-xs"><Link href={`/runs/${run.run_id}`} className="text-blue-600 dark:text-blue-400 hover:underline">{run.run_id}</Link></td>
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
