import RunsTable from "@/components/RunsTable";
import ApprovalsList from "@/components/ApprovalsList";
import ActionChart from "@/components/ActionChart";
import ErrorSummary from "@/components/ErrorSummary";

export default function Dashboard() {
  return (
    <main className="max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">decide-hub Operator Dashboard</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border rounded-lg p-4 shadow-sm">
          <RunsTable />
        </div>
        <div className="bg-white border rounded-lg p-4 shadow-sm">
          <ApprovalsList />
        </div>
        <div className="bg-white border rounded-lg p-4 shadow-sm">
          <ActionChart />
        </div>
        <div className="bg-white border rounded-lg p-4 shadow-sm">
          <ErrorSummary />
        </div>
      </div>
    </main>
  );
}
