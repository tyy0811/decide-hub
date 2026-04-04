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
  const [acting, setActing] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchApprovals = () => {
    fetch(`${API_BASE}/approvals`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => setApprovals(data.approvals ?? []))
      .catch(() => setApprovals([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchApprovals();
  }, []);

  const handleAction = async (id: number, action: "approve" | "reject") => {
    setActing(id);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/approvals/${id}/${action}`, {
        method: "POST",
      });
      if (!resp.ok) throw new Error(`Failed to ${action} (${resp.status})`);
      fetchApprovals();
    } catch (e) {
      setError(e instanceof Error ? e.message : `Failed to ${action}`);
      fetchApprovals();
    } finally {
      setActing(null);
    }
  };

  if (loading) return <div className="text-slate-400">Loading approvals...</div>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1 text-gray-900 dark:text-white">
        Pending Approvals
        {approvals.length > 0 && (
          <span className="ml-2 text-sm bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300 px-2 py-0.5 rounded">
            {approvals.length}
          </span>
        )}
      </h2>
      <p className="text-xs text-slate-400 mb-3">Actions that require human review before execution. Triggered by permission rules.</p>
      {error && (
        <p className="text-xs text-red-600 dark:text-red-400 mb-2">{error}</p>
      )}
      <div className="space-y-2">
        {approvals.map((a) => (
          <div
            key={a.id}
            className="border border-orange-200 dark:border-orange-800 rounded p-3 bg-orange-50 dark:bg-orange-950"
          >
            <div className="flex justify-between items-start">
              <div>
                <span className="font-mono text-xs text-gray-700 dark:text-slate-300">{a.entity_id}</span>
                <span className="mx-2 text-slate-400">&rarr;</span>
                <span className="font-medium text-gray-900 dark:text-white">{a.proposed_action}</span>
              </div>
              <div className="flex gap-2">
                {a.status === "pending" && (
                  <>
                    <button
                      onClick={() => handleAction(a.id, "approve")}
                      disabled={acting === a.id}
                      className="text-xs px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700 disabled:opacity-50"
                      data-testid={`approve-${a.id}`}
                    >
                      {acting === a.id ? "..." : "Approve"}
                    </button>
                    <button
                      onClick={() => handleAction(a.id, "reject")}
                      disabled={acting === a.id}
                      className="text-xs px-3 py-1 rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
                      data-testid={`reject-${a.id}`}
                    >
                      {acting === a.id ? "..." : "Reject"}
                    </button>
                  </>
                )}
                {a.status !== "pending" && (
                  <span className="text-xs bg-orange-200 text-orange-800 dark:bg-orange-900 dark:text-orange-300 px-2 py-0.5 rounded">
                    {a.status}
                  </span>
                )}
              </div>
            </div>
            {a.reason && (
              <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">{a.reason}</p>
            )}
          </div>
        ))}
        {approvals.length === 0 && (
          <p className="text-slate-400 text-sm">No pending approvals</p>
        )}
      </div>
    </div>
  );
}
