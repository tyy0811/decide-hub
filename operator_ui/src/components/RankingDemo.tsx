"use client";

import { useState } from "react";

interface ScoredItem {
  item_id: number;
  score: number;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function RankingDemo() {
  const [userId, setUserId] = useState(42);
  const [policy, setPolicy] = useState("popularity");
  const [items, setItems] = useState<ScoredItem[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchRanking = () => {
    setLoading(true);
    fetch(`${API_BASE}/rank`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, k: 10, policy }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => setItems(data.items ?? []))
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  };

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1 text-gray-900 dark:text-white">Ranking Demo</h2>
      <p className="text-xs text-slate-400 mb-3">Get top-10 movie recommendations for any user. Compare popularity baseline vs learned LightGBM scorer.</p>
      <div className="flex gap-2 mb-3">
        <input
          type="number"
          value={userId}
          onChange={(e) => setUserId(Number(e.target.value))}
          className="w-24 px-2 py-1 rounded border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-gray-900 dark:text-white text-sm"
          placeholder="User ID"
        />
        <select
          value={policy}
          onChange={(e) => setPolicy(e.target.value)}
          className="px-2 py-1 rounded border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-gray-900 dark:text-white text-sm"
        >
          <option value="popularity">popularity</option>
          <option value="scorer">scorer</option>
        </select>
        <button
          onClick={fetchRanking}
          disabled={loading}
          className="px-3 py-1 rounded bg-blue-600 text-white text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "..." : "Rank"}
        </button>
      </div>
      {items.length > 0 && (
        <div className="space-y-1">
          {items.map((item, i) => (
            <div key={item.item_id} className="flex items-center gap-2 text-sm">
              <span className="w-6 text-right text-slate-400 font-mono">{i + 1}.</span>
              <span className="font-mono text-gray-800 dark:text-slate-200">Item {item.item_id}</span>
              <span className="text-xs text-slate-400 ml-auto">{item.score.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
      {items.length === 0 && !loading && (
        <p className="text-slate-400 text-sm">Click Rank to get recommendations</p>
      )}
    </div>
  );
}
