import HealthStatus from "@/components/HealthStatus";
import RankingDemo from "@/components/RankingDemo";
import EvalMetrics from "@/components/EvalMetrics";
import RunsTable from "@/components/RunsTable";
import ApprovalsList from "@/components/ApprovalsList";
import ActionChart from "@/components/ActionChart";
import ErrorSummary from "@/components/ErrorSummary";
import ShadowComparison from "@/components/ShadowComparison";
import AnomalyIndicator from "@/components/AnomalyIndicator";

const card = "bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-lg";

export default function Dashboard() {
  return (
    <main className="max-w-7xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-8">decide-hub Operator Dashboard</h1>

      {/* Top row: health + ranking + eval */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className={card}>
          <HealthStatus />
        </div>
        <div className={card}>
          <RankingDemo />
        </div>
        <div className={card}>
          <EvalMetrics />
        </div>
      </div>

      {/* Bottom grid: automation panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className={card}>
          <RunsTable />
        </div>
        <div className={card}>
          <ApprovalsList />
        </div>
        <div className={card}>
          <ActionChart />
        </div>
        <div className={card}>
          <ErrorSummary />
        </div>
        <div className={card}>
          <ShadowComparison />
        </div>
        <div className={card}>
          <AnomalyIndicator />
        </div>
      </div>
    </main>
  );
}
