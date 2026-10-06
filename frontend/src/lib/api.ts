import type { Source } from "../types";

export interface AppConfig {
  model: string;
  llm_mode: "gemini" | "mock";
  llm_configured: boolean;
  max_message_chars: number;
}

export interface KbDocument {
  doc_id: string;
  title: string;
  doc_type: string;
  status: "active" | "deprecated";
  updated: string;
  owner: string;
  chunks: number;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}`);
  return (await res.json()) as T;
}

export const fetchConfig = () => getJson<AppConfig>("/api/config");
export const fetchDocuments = () => getJson<{ documents: KbDocument[] }>("/api/sources").then((r) => r.documents);

export async function fetchChunk(chunkId: string): Promise<Source> {
  const c = await getJson<{ chunk_id: string; title: string; heading_path: string; status: Source["status"]; owner: string; text: string }>(
    `/api/chunks?chunk_id=${encodeURIComponent(chunkId)}`,
  );
  return { chunk_id: c.chunk_id, document: c.title, section: c.heading_path, status: c.status, owner: c.owner, text: c.text };
}

export interface SessionSummary {
  id: string;
  title: string;
  updated_at: number; // epoch seconds
  language: "en" | "vi";
}

export const fetchSessions = () => getJson<{ sessions: SessionSummary[] }>("/api/sessions").then((r) => r.sessions);

export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`${res.status}`);
}
