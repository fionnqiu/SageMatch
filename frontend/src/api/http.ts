/** Shared fetch helpers. Callers pass a path; JSON vs multipart is chosen here. */

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    try {
      const parsed = JSON.parse(text) as { detail?: string };
      throw new Error(parsed.detail || text || res.statusText);
    } catch (err) {
      if (err instanceof Error && err.message !== text) throw err;
      throw new Error(text || res.statusText);
    }
  }
  if (res.status === 204) return undefined as T;
  const ctype = res.headers.get("content-type") || "";
  if (!ctype.includes("json")) return (await res.text()) as T;
  return res.json() as Promise<T>;
}

export async function sendFile<T>(path: string, file: File, extra?: Record<string, string>): Promise<T> {
  const body = new FormData();
  body.append("file", file);
  if (extra) {
    for (const [k, v] of Object.entries(extra)) body.append(k, v);
  }
  const res = await fetch(path, { method: "POST", body });
  if (!res.ok) {
    const text = await res.text();
    try {
      const parsed = JSON.parse(text) as { detail?: string };
      throw new Error(parsed.detail || text || res.statusText);
    } catch (err) {
      if (err instanceof Error && err.message !== text) throw err;
      throw new Error(text || res.statusText);
    }
  }
  return res.json() as Promise<T>;
}
