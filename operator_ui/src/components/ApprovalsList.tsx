"use client";

import { useEffect, useState } from "react";

interface Approval {
  id: number;
  entity_id: string;
  proposed_action: string;
  reason: string | null;
  status: string;
  created_at: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ApprovalsList() {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/approvals`)
      .then((r) => r.json())
      .then((data) => setApprovals(data.approvals))
      .catch(() => setApprovals([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500">Loading approvals...</div>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-3">
        Pending Approvals
        {approvals.length > 0 && (
          <span className="ml-2 text-sm bg-orange-100 text-orange-800 px-2 py-0.5 rounded">
            {approvals.length}
          </span>
        )}
      </h2>
      <div className="space-y-2">
        {approvals.map((a) => (
          <div
            key={a.id}
            className="border border-orange-200 rounded p-3 bg-orange-50"
          >
            <div className="flex justify-between items-start">
              <div>
                <span className="font-mono text-xs">{a.entity_id}</span>
                <span className="mx-2 text-gray-400">&rarr;</span>
                <span className="font-medium">{a.proposed_action}</span>
              </div>
              <span className="text-xs bg-orange-200 text-orange-800 px-2 py-0.5 rounded">
                {a.status}
              </span>
            </div>
            {a.reason && (
              <p className="text-xs text-gray-500 mt-1">{a.reason}</p>
            )}
          </div>
        ))}
        {approvals.length === 0 && (
          <p className="text-gray-400 text-sm">No pending approvals</p>
        )}
      </div>
    </div>
  );
}
