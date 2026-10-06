import type { ChatMessage, Connection, HistoryItem, ServerEvent, Source, ToolStep } from "../types";

export interface ChatState {
  sessionId: string | null;
  connection: Connection;
  messages: ChatMessage[];
  sources: Record<string, Source>;
  busy: boolean;
  model: string;
  llmMode: "gemini" | "mock" | null;
}

export const initialState: ChatState = {
  sessionId: null,
  connection: "connecting",
  messages: [],
  sources: {},
  busy: false,
  model: "",
  llmMode: null,
};

export type Action =
  | { kind: "server"; event: ServerEvent }
  | { kind: "connection"; connection: Connection }
  | { kind: "user_send"; id: string; text: string }
  | { kind: "reset" };

let counter = 0;
const uid = (p: string) => `${p}-${Date.now().toString(36)}-${counter++}`;

const fromHistory = (h: HistoryItem): ChatMessage => ({
  id: uid(h.role),
  role: h.role,
  text: h.text,
  status: "done",
  steps: [],
  guardrails: h.guardrails ?? null,
});

/** Apply `fn` to the assistant message currently being generated (the last one). */
function updateLast(state: ChatState, fn: (m: ChatMessage) => ChatMessage): ChatState {
  const idx = state.messages.length - 1;
  if (idx < 0 || state.messages[idx].role !== "assistant") return state;
  const messages = state.messages.slice();
  messages[idx] = fn(messages[idx]);
  return { ...state, messages };
}

export function chatReducer(state: ChatState, action: Action): ChatState {
  switch (action.kind) {
    case "connection": {
      let next: ChatState = { ...state, connection: action.connection };
      // A dropped socket mid-answer must not leave the UI spinning forever.
      if (action.connection === "closed" && state.busy) {
        next = updateLast({ ...next, busy: false }, (m) => ({
          ...m,
          status: "error",
          error: { code: "CONNECTION_LOST", message: "Connection lost." },
        }));
      }
      return next;
    }
    case "user_send":
      return {
        ...state,
        busy: true,
        messages: [
          ...state.messages,
          { id: action.id, role: "user", text: action.text, status: "done", steps: [] },
          { id: uid("assistant"), role: "assistant", text: "", status: "streaming", steps: [] },
        ],
      };
    case "reset":
      return { ...initialState, connection: state.connection, model: state.model, llmMode: state.llmMode };
    case "server":
      return applyServerEvent(state, action.event);
  }
}

function applyServerEvent(state: ChatState, ev: ServerEvent): ChatState {
  switch (ev.type) {
    case "session":
      return {
        ...state,
        sessionId: ev.session_id,
        model: ev.model,
        llmMode: ev.llm_mode,
        // Only restore the transcript when we are not mid-conversation (page reload / reconnect).
        messages: state.busy ? state.messages : ev.history.map(fromHistory),
      };
    case "token":
      return updateLast(state, (m) => ({ ...m, text: m.text + ev.delta }));
    case "tool_call": {
      const step: ToolStep = { id: ev.id, name: ev.name, args: ev.args, status: "running" };
      // Text streamed before a tool call is narration, not the answer: reset the live buffer.
      return updateLast(state, (m) => ({ ...m, text: "", steps: [...m.steps, step] }));
    }
    case "tool_result":
      return updateLast(state, (m) => ({
        ...m,
        steps: m.steps.map((s) =>
          s.id === ev.id || (s.status === "running" && s.name === ev.name && !ev.id)
            ? { ...s, status: ev.status, summary: ev.summary, downloadPath: ev.download_path }
            : s,
        ),
      }));
    case "sources": {
      const sources = { ...state.sources };
      for (const c of ev.chunks) sources[c.chunk_id] = c;
      return { ...state, sources };
    }
    case "final":
      // The server's final text is authoritative: it replaces whatever was streamed (the
      // compliance guardrail may have edited it).
      return updateLast(state, (m) => ({ ...m, text: ev.text, guardrails: ev.guardrails, status: "done" }));
    case "error":
      return updateLast(state, (m) => ({ ...m, status: "error", error: { code: ev.code, message: ev.message } }));
    case "turn_end":
      return updateLast({ ...state, busy: false }, (m) => ({
        ...m,
        usage: ev.usage,
        status: m.status === "streaming" ? "done" : m.status,
      }));
    case "turn_start":
    case "pong":
      return state;
  }
}
