import { test, expect } from "@playwright/test";

const API_BASE = "http://localhost:8000";
const MOCK_SOURCE = "http://localhost:9999/leads";

test.describe("Approval Actions E2E", () => {
  test("approve a pending approval and verify execution", async ({
    page,
    request,
  }) => {
    // Step 1: Trigger automation to create a pending approval
    const automateResponse = await request.post(`${API_BASE}/automate`, {
      data: { source_url: MOCK_SOURCE, dry_run: false },
    });
    expect(automateResponse.ok()).toBeTruthy();

    // Step 2: Get pending approvals via API
    const approvalsResponse = await request.get(`${API_BASE}/approvals`);
    const approvalsData = await approvalsResponse.json();
    const pendingApproval = approvalsData.approvals.find(
      (a: { proposed_action: string; status: string }) =>
        a.proposed_action === "send_external_email" && a.status === "pending"
    );
    expect(pendingApproval).toBeDefined();

    // Step 3: Load dashboard
    await page.goto("/");
    await expect(
      page.getByText("send_external_email", { exact: true }).first()
    ).toBeVisible();

    // Step 4: Click Approve button
    await page
      .getByRole("button", { name: "Approve" })
      .first()
      .click();

    // Step 5: Verify approval disappears from pending list
    // Wait for refetch
    await page.waitForTimeout(1000);
    const remainingApprovals = await request.get(`${API_BASE}/approvals`);
    const remainingData = await remainingApprovals.json();
    const stillPending = remainingData.approvals.filter(
      (a: { id: number; status: string }) =>
        a.id === pendingApproval.id && a.status === "pending"
    );
    expect(stillPending.length).toBe(0);
  });

  test("reject a pending approval and verify it does not execute", async ({
    request,
  }) => {
    // Step 1: Trigger automation
    await request.post(`${API_BASE}/automate`, {
      data: { source_url: MOCK_SOURCE, dry_run: false },
    });

    // Step 2: Get and reject the approval via API
    const approvalsResponse = await request.get(`${API_BASE}/approvals`);
    const approvalsData = await approvalsResponse.json();
    const pendingApproval = approvalsData.approvals.find(
      (a: { proposed_action: string; status: string }) =>
        a.proposed_action === "send_external_email" && a.status === "pending"
    );
    expect(pendingApproval).toBeDefined();

    const rejectResponse = await request.post(
      `${API_BASE}/approvals/${pendingApproval.id}/reject`
    );
    expect(rejectResponse.ok()).toBeTruthy();
    const rejectData = await rejectResponse.json();
    expect(rejectData.status).toBe("rejected");

    // Step 3: Verify it no longer appears in pending
    const updatedApprovals = await request.get(`${API_BASE}/approvals`);
    const updatedData = await updatedApprovals.json();
    const stillPending = updatedData.approvals.filter(
      (a: { id: number }) => a.id === pendingApproval.id
    );
    expect(stillPending.length).toBe(0);
  });
});
