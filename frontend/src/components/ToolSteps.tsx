import type { Language, ToolStep } from "../types";
import { t } from "../i18n";

const OK = new Set(["ok", "saved"]);

function argPreview(step: ToolStep): string {
  const a = step.args;
  if (typeof a.query === "string") return `“${a.query}”`;
  if (typeof a.doc_id === "string") return a.doc_id;
  if (typeof a.title === "string") return a.title;
  if (typeof a.copy_text === "string") return `${a.copy_text.slice(0, 60)}${a.copy_text.length > 60 ? "…" : ""}`;
  return "";
}

export function ToolSteps({ steps, lang, live }: { steps: ToolStep[]; lang: Language; live: boolean }) {
  if (!steps.length) return null;
  const d = t(lang);
  return (
    <details className="steps" open={live}>
      <summary>
        {d.steps} · {steps.length}
      </summary>
      <ol>
        {steps.map((s) => {
          const state = s.status === "running" ? "running" : OK.has(s.status) ? "ok" : "bad";
          return (
            <li key={s.id} className={`step ${state}`}>
              <span className="dot" aria-hidden />
              <span className="step-name">{d.tools[s.name] ?? s.name}</span>
              <span className="step-arg">{argPreview(s)}</span>
              {s.summary && <span className="step-summary">{s.summary}</span>}
              {s.downloadPath && (
                <a className="step-link" href={`${s.downloadPath}?download=true`}>
                  {d.downloadBrief}
                </a>
              )}
            </li>
          );
        })}
      </ol>
    </details>
  );
}
