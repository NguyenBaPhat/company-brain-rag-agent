// Wire protocol — mirrors backend/src/company_brain/api/schemas.py and events.py.

export type Language = "en" | "vi";

export interface BlockedCopy { rule_id: string; match: string; suggestion: string; policy_section: string }

export interface Guardrails {
  citations: { cited: string[]; verified: string[]; unverified: string[] };
  compliance: {
    copy_blocks: number;
    blocked: BlockedCopy[];
    warnings: { rule_id: string; match: string }[];
    missing_disclaimers: string[];
    rules_version: string;
  };
  grounding: { kb_searches: number; best_confidence: "none" | "medium" | "high"; ungrounded: boolean };
  modified: boolean;
}

export interface Source {
  chunk_id: string;
  document: string | null;
  section: string | null;
  status: "active" | "deprecated" | null;
  owner?: string | null;
  relevance?: number | null;
  text: string;
}

export interface Usage { prompt_tokens: number; output_tokens: number; tool_calls: number; latency_ms: number }

export type ServerEvent =
  | { type: "session"; session_id: string; history: HistoryItem[]; model: string; llm_mode: "gemini" | "mock"; languages: Language[] }
  | { type: "turn_start"; message_id: string }
  | { type: "token"; delta: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; id: string; name: string; status: string; summary: string; download_path?: string }
  | { type: "sources"; chunks: Source[] }
  | { type: "final"; text: string; guardrails: Guardrails | null }
  | { type: "error"; code: string; message: string }
  | { type: "turn_end"; message_id: string; usage: Usage }
  | { type: "pong" };

export interface HistoryItem { role: "user" | "assistant"; text: string; guardrails?: Guardrails | null }

export interface ToolStep {
  id: string;
  name: string;
  args: Record<string, unknown>;
  status: "running" | "ok" | "error" | "rejected" | "no_relevant_results" | "not_found" | "saved" | string;
  summary?: string;
  downloadPath?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  status: "streaming" | "done" | "error";
  steps: ToolStep[];
  guardrails?: Guardrails | null;
  error?: { code: string; message: string };
  usage?: Usage;
}

export type Connection = "connecting" | "open" | "closed";
