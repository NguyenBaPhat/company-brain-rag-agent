import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Composer } from "./components/Composer";
import { MessageBubble } from "./components/MessageBubble";
import { Sidebar } from "./components/Sidebar";
import { SidePanel } from "./components/SidePanel";
import { useAgentSocket } from "./hooks/useAgentSocket";
import { detectLanguage, t } from "./i18n";
import { deleteSession, fetchConfig, fetchSessions, type AppConfig, type SessionSummary } from "./lib/api";
import { latestCitations } from "./lib/citations";
import { chatReducer, initialState } from "./state/chatReducer";
import type { Language } from "./types";

export default function App() {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const { send, cancel, newSession, switchSession } = useAgentSocket(dispatch);
  const [lang, setLang] = useState<Language>(detectLanguage);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [tab, setTab] = useState<"sources" | "kb">("sources");
  const [focus, setFocus] = useState<{ ids: string[]; active: string | null }>({ ids: [], active: null });
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const d = t(lang);

  const refreshSessions = useCallback(() => {
    fetchSessions().then(setSessions).catch(() => undefined);
  }, []);
  // Reload the history list on connect/switch and whenever an answer finishes (titles and ordering change).
  useEffect(() => {
    if (!state.busy) refreshSessions();
  }, [state.busy, state.sessionId, refreshSessions]);

  const onDelete = async (id: string) => {
    await deleteSession(id).catch(() => undefined);
    if (id === state.sessionId) newSession();
    refreshSessions();
  };

  useEffect(() => {
    localStorage.setItem("cb.lang", lang);
    document.documentElement.lang = lang;
  }, [lang]);
  useEffect(() => {
    fetchConfig().then(setConfig).catch(() => undefined);
  }, []);

  // The Sources panel always reflects the latest answer of the OPEN conversation. It is reset when a new
  // question is sent (the new answer has no citations yet), when another conversation is opened, and on
  // "New chat"; clicking a chip in an older answer temporarily focuses that answer's sources instead.
  const last = state.messages.at(-1);
  const lastCited = latestCitations(state.messages);
  const focusKey = `${state.sessionId}|${last?.id}|${lastCited.join("|")}`;
  useEffect(() => {
    setFocus({ ids: lastCited, active: null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusKey]);

  // Auto-scroll while streaming, unless the user scrolled up to read.
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  useEffect(() => {
    const el = scroller.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [state.messages]);

  const onCite = (ids: string[], chunkId: string) => {
    setFocus({ ids, active: chunkId });
    setTab("sources");
  };
  const connected = state.connection === "open";
  const mock = (state.llmMode ?? config?.llm_mode) === "mock";
  const empty = state.messages.length === 0;
  const suggestions = useMemo(() => d.suggestions, [d]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <button type="button" className="burger" aria-label={d.toggleSidebar} onClick={() => setSidebarOpen((o) => !o)}>
            ☰
          </button>
          <span className="logo" aria-hidden>L</span>
          <div>
            <h1>{d.appTitle}</h1>
            <p>{d.appSubtitle}</p>
          </div>
        </div>
        <div className="topbar-right">
          {mock && <span className="chip warn" title="LLM_MODE=mock">{d.mockBadge}</span>}
          {!connected && (
            <span className={`conn ${state.connection}`} role="status">
              <span className="dot" /> {d.connection[state.connection]}
            </span>
          )}
          <div className="seg" role="group" aria-label={d.language}>
            {(["en", "vi"] as const).map((l) => (
              <button key={l} className={lang === l ? "on" : ""} aria-pressed={lang === l} onClick={() => setLang(l)}>
                {l.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="layout">
        <Sidebar
          lang={lang}
          sessions={sessions}
          activeId={state.sessionId}
          open={sidebarOpen}
          onNew={newSession}
          onSelect={switchSession}
          onDelete={onDelete}
          onClose={() => setSidebarOpen(false)}
        />
        <section className="chat">
          <div
            className="scroll"
            ref={scroller}
            onScroll={(e) => {
              const el = e.currentTarget;
              pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
            }}
          >
            {empty ? (
              <div className="empty">
                <h2>{d.emptyTitle}</h2>
                <p>{d.emptyBody}</p>
                <div className="try">{d.tryThese}</div>
                <div className="suggestions">
                  {suggestions.map((s) => (
                    <button key={s} disabled={!connected || state.busy} onClick={() => send(s, lang)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="messages">
                {state.messages.map((m) => (
                  <MessageBubble key={m.id} message={m} lang={lang} sources={state.sources} activeCitation={focus.active} onCite={onCite} />
                ))}
              </div>
            )}
          </div>
          <Composer
            lang={lang}
            busy={state.busy}
            disabled={!connected}
            maxChars={config?.max_message_chars ?? 2000}
            onSend={(text) => send(text, lang)}
            onStop={cancel}
          />
        </section>
        <SidePanel lang={lang} tab={tab} onTab={setTab} citationIds={focus.ids} activeCitation={focus.active} sources={state.sources} />
      </main>
    </div>
  );
}
