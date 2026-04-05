import { test, expect } from "@playwright/test";
import { getAuthToken, authHeaders } from "./auth-helpers";

const API_BASE = "http://localhost:8000";
const MOCK_SOURCE = "http://localhost:9999/leads";

test.describe("Automation E2E", () => {
  test("trigger automation, verify API state, then verify dashboard renders data", async ({
    page,
    request,
  }) => {
    const token = await getAuthToken(request);
    const headers = authHeaders(token);

    // Step 1: Verify API is healthy (no auth required)
    const healthResponse = await request.get(`${API_BASE}/health`);
    expect(healthResponse.ok()).toBeTruthy();

    // Step 2: Trigger automation against mock lead source
    const automateResponse = await request.post(`${API_BASE}/automate`, {
      data: {
        source_url: MOCK_SOURCE,
        dry_run: false,
      },
      headers,
    });
    expect(automateResponse.ok()).toBeTruthy();
    const automateData = await automateResponse.json();
    expect(automateData.status).toBe("completed");
    expect(automateData.entities_processed).toBeGreaterThan(0);

    // Step 3: Verify run appears in /runs
    const runsResponse = await request.get(`${API_BASE}/runs`, { headers });
    expect(runsResponse.ok()).toBeTruthy();
    const runsData = await runsResponse.json();
    expect(runsData.runs.length).toBeGreaterThan(0);
    const matchingRun = runsData.runs.find(
      (r: { run_id: string }) => r.run_id === automateData.run_id
    );
    expect(matchingRun).toBeDefined();
    expect(matchingRun.status).toBe("completed");

    // Step 4: Verify approval was queued (mock data includes request_email lead)
    const approvalsResponse = await request.get(`${API_BASE}/approvals`, { headers });
    expect(approvalsResponse.ok()).toBeTruthy();
    const approvalsData = await approvalsResponse.json();
    const emailApprovals = approvalsData.approvals.filter(
      (a: { proposed_action: string }) => a.proposed_action === "send_external_email"
    );
    expect(emailApprovals.length).toBeGreaterThan(0);

    // Step 5: Load dashboard — log in via UI first
    await page.goto("/login");
    await page.fill('input[placeholder="Username"]', "admin");
    await page.fill('input[placeholder="Password"]', "admin");
    await page.click('button[type="submit"]');
    await page.waitForURL("/");

    // Headings present
    await expect(page.getByRole("heading", { name: "decide-hub Operator Dashboard" })).toBeVisible();

    // Runs table shows the run ID we just created
    await expect(page.getByText(automateData.run_id)).toBeVisible();

    // Approvals section shows the queued approval action
    await expect(page.getByText("send_external_email", { exact: true }).first()).toBeVisible();

    // Action distribution section rendered
    await expect(page.getByRole("heading", { name: "Action Distribution" })).toBeVisible();
  });
});
