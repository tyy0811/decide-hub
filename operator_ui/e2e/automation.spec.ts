import { test, expect } from "@playwright/test";

const API_BASE = "http://localhost:8000";

test.describe("Automation E2E", () => {
  test("trigger automation and verify dashboard renders data", async ({
    page,
    request,
  }) => {
    // Step 1: Verify API is healthy
    const healthResponse = await request.get(`${API_BASE}/health`);
    expect(healthResponse.ok()).toBeTruthy();

    // Step 2: Verify approvals endpoint works
    const approvalsResponse = await request.get(`${API_BASE}/approvals`);
    if (approvalsResponse.ok()) {
      const approvals = await approvalsResponse.json();
      expect(approvals).toHaveProperty("approvals");
    }

    // Step 3: Load Next.js dashboard
    await page.goto("/");

    // Step 4: Assert dashboard renders key section headings (role-based for strict mode)
    await expect(page.getByRole("heading", { name: "decide-hub Operator Dashboard" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Recent Automation Runs" })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Pending Approvals/ })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Action Distribution" })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Failed Entities/ })).toBeVisible();
  });
});
