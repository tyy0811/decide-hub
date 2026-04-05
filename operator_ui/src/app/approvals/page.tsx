import ApprovalsList from "@/components/ApprovalsList";

export default function ApprovalsPage() {
  return (
    <main className="max-w-7xl mx-auto p-8">
      <h1 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">Approval Queue</h1>
      <ApprovalsList />
    </main>
  );
}
