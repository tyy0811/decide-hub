"use client";

import { useEffect, useState } from "react";

interface Health {
  status: string;
  policies: string[];
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function HealthStatus() {
  const [health, setHealth] = useState<Health | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => setHealth(data))
      .catch(() => setHealth(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-slate-400">Checking health...</div>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1 text-gray-900 dark:text-white">System Health</h2>
      <p className="text-xs text-slate-400 mb-3">API status and which ranking policies are loaded and ready to serve.</p>
      {health ? (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-green-500 inline-block"></span>
            <span className="text-gray-800 dark:text-slate-200 font-medium">API Online</span>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-slate-400 mb-2">Loaded Policies:</p>
            <div className="flex flex-wrap gap-2">
              {health.policies.map((p) => (
                <span
                  key={p}
                  className="px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300"
                >
                  {p}
                </span>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-red-500 inline-block"></span>
          <span className="text-red-600 dark:text-red-400 font-medium">API Offline</span>
        </div>
      )}
    </div>
  );
}
