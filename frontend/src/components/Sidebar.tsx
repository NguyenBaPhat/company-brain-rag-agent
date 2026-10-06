import { t } from "../i18n";
import type { SessionSummary } from "../lib/api";
import { groupSessions } from "../lib/sessions";
import type { Language } from "../types";

interface Props {
  lang: Language;
  sessions: SessionSummary[];
  activeId: string | null;
  open: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

export function Sidebar({ lang, sessions, activeId, open, onNew, onSelect, onDelete, onClose }: Props) {
  const d = t(lang);
  const groups = groupSessions(sessions);
  return (
    <>
      {open && <div className="scrim" onClick={onClose} aria-hidden />}
      <nav className={`sidebar ${open ? "open" : ""}`} aria-label={d.history}>
        <button type="button" className="btn new-chat" onClick={() => { onNew(); onClose(); }}>
          + {d.newChat}
        </button>
        <div className="history">
          {groups.length === 0 && <p className="empty-note">{d.noHistory}</p>}
          {groups.map(({ bucket, items }) => (
            <section key={bucket}>
              <h3>{d.buckets[bucket]}</h3>
              <ul>
                {items.map((s) => (
                  <li key={s.id} className={s.id === activeId ? "active" : ""}>
                    <button type="button" className="item" title={s.title} onClick={() => { onSelect(s.id); onClose(); }}>
                      {s.title}
                    </button>
                    <button
                      type="button"
                      className="del"
                      aria-label={d.deleteChat}
                      title={d.deleteChat}
                      onClick={() => window.confirm(d.deleteConfirm) && onDelete(s.id)}
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </nav>
    </>
  );
}
