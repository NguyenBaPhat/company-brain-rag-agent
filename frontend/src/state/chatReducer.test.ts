import { describe, expect, it } from "vitest";
import { chatReducer, initialState, type Action, type ChatState } from "./chatReducer";
import type { ServerEvent } from "../types";

const run = (events: Action[], start: ChatState = initialState) => events.reduce(chatReducer, start);
const srv = (event: ServerEvent): Action => ({ kind: "server", event });
const send: Action = { kind: "user_send", id: "u1", text: "hello" };

describe("chatReducer", () => {
  it("streams tokens, then the final text replaces the streamed text", () => {
    const s = run([
      send,
      srv({ type: "token", delta: "Draft " }),
      srv({ type: "token", delta: "text" }),
      srv({ type: "final", text: "Edited by guardrail", guardrails: null }),
      srv({ type: "turn_end", message_id: "m", usage: { prompt_tokens: 1, output_tokens: 2, tool_calls: 0, latency_ms: 5 } }),
    ]);
    const last = s.messages.at(-1)!;
    expect(last.text).toBe("Edited by guardrail");
    expect(last.status).toBe("done");
    expect(s.busy).toBe(false);
    expect(last.usage?.latency_ms).toBe(5);
  });

  it("clears narration streamed before a tool call and tracks the step lifecycle", () => {
    const s = run([
      send,
      srv({ type: "token", delta: "Let me search..." }),
      srv({ type: "tool_call", id: "c1", name: "search_knowledge_base", args: { query: "price" } }),
    ]);
    expect(s.messages.at(-1)!.text).toBe("");
    expect(s.messages.at(-1)!.steps[0].status).toBe("running");

    const done = run([srv({ type: "tool_result", id: "c1", name: "search_knowledge_base", status: "ok", summary: "5 passages" })], s);
    expect(done.messages.at(-1)!.steps[0]).toMatchObject({ status: "ok", summary: "5 passages" });
  });

  it("indexes retrieved sources by chunk id", () => {
    const s = run([srv({ type: "sources", chunks: [{ chunk_id: "a#b", document: "D", section: "S", status: "active", text: "t" }] })]);
    expect(s.sources["a#b"].document).toBe("D");
  });

  it("restores history on session event but never clobbers an in-flight turn", () => {
    const history = [{ role: "user" as const, text: "q" }, { role: "assistant" as const, text: "a" }];
    const session = srv({ type: "session", session_id: "s", history, model: "m", llm_mode: "mock", languages: ["en", "vi"] });
    expect(run([session]).messages).toHaveLength(2);
    const busy = run([send, session]);
    expect(busy.messages).toHaveLength(2); // the live user+assistant pair, not the restored history
    expect(busy.messages[0].text).toBe("hello");
  });

  it("surfaces errors on the assistant message and stops the spinner on turn_end", () => {
    const s = run([send, srv({ type: "error", code: "LLM_AUTH", message: "bad key" }), srv({ type: "turn_end", message_id: "m", usage: { prompt_tokens: 0, output_tokens: 0, tool_calls: 0, latency_ms: 1 } })]);
    expect(s.messages.at(-1)).toMatchObject({ status: "error", error: { code: "LLM_AUTH" } });
    expect(s.busy).toBe(false);
  });

  it("marks the in-flight answer as failed when the socket drops", () => {
    const s = run([send, { kind: "connection", connection: "closed" }]);
    expect(s.busy).toBe(false);
    expect(s.messages.at(-1)!.error?.code).toBe("CONNECTION_LOST");
  });
});
