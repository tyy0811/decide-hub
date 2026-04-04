import { defineConfig } from "@playwright/test";
import { existsSync } from "fs";
import { resolve } from "path";

// Use .venv/bin/python if it exists (local dev), else fall back to python3 (CI)
const projectRoot = resolve(__dirname, "..");
const venvPython = resolve(projectRoot, ".venv/bin/python");
const python = existsSync(venvPython) ? venvPython : "python3";

// E2E tests live in operator_ui/e2e/ (colocated with the UI they test)
// rather than tests/e2e/ — keeps Playwright config paths simple and
// groups frontend tests with frontend code.
export default defineConfig({
  testDir: "./e2e",
  timeout: 60000,
  retries: 1,
  use: {
    baseURL: "http://localhost:3000",
    headless: true,
  },
  webServer: [
    {
      command: `${python} ${resolve(__dirname, "e2e/mock_server.py")}`,
      port: 9999,
      timeout: 10000,
      reuseExistingServer: true,
    },
    {
      command: `cd ${projectRoot} && ${python} -m uvicorn src.serving.app:app --port 8000`,
      port: 8000,
      timeout: 120000,
      reuseExistingServer: true,
    },
    {
      command: "npm run dev",
      port: 3000,
      timeout: 30000,
      reuseExistingServer: true,
    },
  ],
});
