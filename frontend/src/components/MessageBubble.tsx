import { memo } from "react";
import { t } from "../i18n";
import type { ChatMessage, Language, Source } from "../types";
import { GuardrailBar } from "./GuardrailBar";
import { Markdown } from "./Markdown";
import { ToolSteps } from "./ToolSteps";

interface Props {
  message: ChatMessage;
  lang: Language;
  sources: Record<string, Source>;
  activeCitation: string | null;
  onCite: (messageCitations: string[], chunkId: string) => void;
}

export const MessageBubble = memo(function MessageBubble({ message: m, lang, sources, activeCitation, onCite }: Props) {
  const d = t(lang);
  if (m.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{m.text}</div>
      </div>
    );
  }
  const live = m.status === "streaming";
  const cited = m.guardrails?.citations.cited ?? [];
  return (
    <div className="msg assistant">
      <div className="avatar" aria-hidden>
        L
      </div>
      <div className="body">
        <div className="who">{d.assistant}</div>
        <ToolSteps steps={m.steps} lang={lang} live={live} />
        {m.text ? (
          <Markdown text={m.text} sources={sources} activeCitation={activeCitation} onCite={(id) => onCite(cited, id)} />
        ) : (
          live && <span className="typing">{d.thinking}</span>
        )}
        {live && m.text && <span className="caret" aria-hidden />}
        {m.error && (
          <div className="error-banner" role="alert">
            {d.errors[m.error.code] ?? m.error.message}
          </div>
        )}
        {m.guardrails && <GuardrailBar guardrails={m.guardrails} lang={lang} />}
      </div>
    </div>
  );
});
