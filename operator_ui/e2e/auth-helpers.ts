/**
 * Shared auth helpers for E2E tests.
 * Logs in as the admin demo user and returns a token for API calls.
 */

const API_BASE = "http://localhost:8000";

export async function getAuthToken(request: {
  post: (url: string, options?: object) => Promise<{ json: () => Promise<{ token: string }> }>;
}): Promise<string> {
  const resp = await request.post(`${API_BASE}/auth/login`, {
    data: { username: "admin", password: "admin" },
  });
  const data = await resp.json();
  return data.token;
}

export function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}
