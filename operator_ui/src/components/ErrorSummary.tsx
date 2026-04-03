"use client";

import { useEffect, useState } from "react";

interface FailedEntity {
  entity_id: string;
  run_id: string;
  error_type: string;
  error_message: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ErrorSummary() {
  const [errors, setErrors] = useState<FailedEntity[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/failed-entities`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => setErrors(data.failed_entities ?? []))
      .catch(() => setErrors([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500">Loading errors...</div>;

  // Group by error_type
  const grouped = errors.reduce<Record<string, FailedEntity[]>>((acc, e) => {
    acc[e.error_type] = acc[e.error_type] || [];
    acc[e.error_type].push(e);
    return acc;
  }, {});

  return (
    <div>
      <h2 className="text-lg font-semibold mb-3">
        Failed Entities
        {errors.length > 0 && (
          <span className="ml-2 text-sm bg-red-100 text-red-800 px-2 py-0.5 rounded">
            {errors.length}
          </span>
        )}
      </h2>
      {Object.keys(grouped).length === 0 ? (
        <p className="text-gray-400 text-sm">No failures</p>
      ) : (
        <div className="space-y-3">
          {Object.entries(grouped).map(([errorType, entities]) => (
            <div key={errorType} className="border border-red-200 rounded p-3">
              <div className="flex justify-between items-center mb-1">
                <span className="font-mono text-sm text-red-700">
                  {errorType}
                </span>
                <span className="text-xs bg-red-100 text-red-800 px-2 py-0.5 rounded">
                  {entities.length}
                </span>
              </div>
              <ul className="text-xs text-gray-600 space-y-1">
                {entities.slice(0, 3).map((e) => (
                  <li key={e.entity_id}>
                    <span className="font-mono">{e.entity_id}</span>:{" "}
                    {e.error_message}
                  </li>
                ))}
                {entities.length > 3 && (
                  <li className="text-gray-400">
                    +{entities.length - 3} more
                  </li>
                )}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
