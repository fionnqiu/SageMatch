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

export async function readEventStream(
  path: string,
  body: unknown,
  onEvent: (event: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<void> {
  // 不用 EventSource。发送要带 JSON，而且这一轮不能在断线后自动重放。
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text();
    try {
      const parsed = JSON.parse(text) as { detail?: string };
      throw new Error(parsed.detail || text || res.statusText);
    } catch (err) {
      if (err instanceof Error && err.message !== text) throw err;
      throw new Error(text || res.statusText);
    }
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (!data) continue;
      onEvent(JSON.parse(data) as Record<string, unknown>);
    }
  }
  // 连接被掐断时，最后一帧可能还留在缓冲区里。丢掉就会把已生成的结果当成失败。
  const tail = buffer.trim();
  if (!tail.startsWith("data:")) return;
  const data = tail
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("\n");
  if (data) onEvent(JSON.parse(data) as Record<string, unknown>);
}
