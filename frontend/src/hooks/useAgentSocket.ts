import { useCallback, useEffect, useRef, type Dispatch } from "react";
import type { Action } from "../state/chatReducer";
import type { Language, ServerEvent } from "../types";

const SESSION_KEY = "cb.session";
const PING_MS = 25_000;
const MAX_BACKOFF_MS = 8_000;

const newSessionId = () =>
  (globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`).replace(/-/g, "");

function wsUrl(sessionId: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const token = import.meta.env.VITE_WS_TOKEN as string | undefined;
  const qs = new URLSearchParams({ session_id: sessionId, ...(token ? { token } : {}) });
  return `${proto}://${location.host}/ws/chat?${qs}`;
}

/**
 * Owns the WebSocket: connect, ping keep-alive, exponential-backoff reconnect, and a small API
 * (send / cancel / newSession). All server events are dispatched into the chat reducer.
 */
export function useAgentSocket(dispatch: Dispatch<Action>) {
  const wsRef = useRef<WebSocket | null>(null);
  const sessionRef = useRef<string>(localStorage.getItem(SESSION_KEY) ?? newSessionId());
  const stoppedRef = useRef(false);
  const attemptRef = useRef(0);
  const timersRef = useRef<{ retry?: number; ping?: number }>({});

  const connect = useCallback(() => {
    if (stoppedRef.current) return;
    dispatch({ kind: "connection", connection: "connecting" });
    const ws = new WebSocket(wsUrl(sessionRef.current));
    wsRef.current = ws;

    ws.onopen = () => {
      attemptRef.current = 0;
      dispatch({ kind: "connection", connection: "open" });
      timersRef.current.ping = window.setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
      }, PING_MS);
    };
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data as string) as ServerEvent;
        if (event.type === "session") {
          sessionRef.current = event.session_id;
          localStorage.setItem(SESSION_KEY, event.session_id);
        }
        dispatch({ kind: "server", event });
      } catch {
        /* ignore malformed frames */
      }
    };
    ws.onclose = () => {
      window.clearInterval(timersRef.current.ping);
      if (wsRef.current === ws) wsRef.current = null;
      dispatch({ kind: "connection", connection: "closed" });
      if (stoppedRef.current) return;
      const delay = Math.min(MAX_BACKOFF_MS, 500 * 2 ** attemptRef.current++);
      timersRef.current.retry = window.setTimeout(connect, delay);
    };
    ws.onerror = () => ws.close();
  }, [dispatch]);

  useEffect(() => {
    stoppedRef.current = false;
    connect();
    return () => {
      stoppedRef.current = true;
      window.clearTimeout(timersRef.current.retry);
      window.clearInterval(timersRef.current.ping);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback(
    (text: string, language: Language): boolean => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !text.trim()) return false;
      const id = newSessionId().slice(0, 12);
      dispatch({ kind: "user_send", id, text });
      ws.send(JSON.stringify({ type: "user_message", text, language, message_id: id }));
      return true;
    },
    [dispatch],
  );

  const cancel = useCallback(() => {
    wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.send(JSON.stringify({ type: "cancel" }));
  }, []);

  /** Point the socket at another conversation: clear the UI, reconnect, server restores the history. */
  const switchSession = useCallback(
    (id: string) => {
      if (id === sessionRef.current) return;
      sessionRef.current = id;
      localStorage.setItem(SESSION_KEY, id);
      dispatch({ kind: "reset" });
      wsRef.current?.close(); // onclose reconnects with the new session id
    },
    [dispatch],
  );

  /** Start a fresh, empty conversation. */
  const newSession = useCallback(() => {
    switchSession(newSessionId());
  }, [switchSession]);

  return { send, cancel, newSession, switchSession };
}
