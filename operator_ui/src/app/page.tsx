import RunsTable from "@/components/RunsTable";
import ApprovalsList from "@/components/ApprovalsList";
import ActionChart from "@/components/ActionChart";
import ErrorSummary from "@/components/ErrorSummary";

export default function Dashboard() {
  return (
    <main className="max-w-6xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-8">decide-hub Operator Dashboard</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-lg">
          <RunsTable />
        </div>
        <div className="bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-lg">
          <ApprovalsList />
        </div>
        <div className="bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-lg">
          <ActionChart />
        </div>
        <div className="bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-lg">
          <ErrorSummary />
        </div>
      </div>
    </main>
  );
}
