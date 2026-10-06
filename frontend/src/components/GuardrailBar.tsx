import type { Guardrails, Language } from "../types";
import { t } from "../i18n";

type Tone = "good" | "warn" | "bad" | "info";
interface Pill { tone: Tone; label: string; title?: string }

export function guardrailPills(g: Guardrails, lang: Language): Pill[] {
  const d = t(lang).guard;
  const pills: Pill[] = [];
  const { cited, verified, unverified } = g.citations;

  if (cited.length) pills.push({ tone: unverified.length ? "bad" : "good", label: d.verified(verified.length, cited.length), title: verified.join("\n") });
  if (unverified.length) pills.push({ tone: "bad", label: d.unverified(unverified.length), title: unverified.join("\n") });
  if (!cited.length && g.grounding.kb_searches > 0) {
    pills.push({ tone: g.grounding.best_confidence === "none" ? "info" : "warn", label: d.noCitations });
  }
  if (g.grounding.kb_searches > 0) {
    const c = g.grounding.best_confidence;
    pills.push({ tone: c === "high" ? "good" : c === "medium" ? "warn" : "info", label: d.evidence[c] });
  }

  const c = g.compliance;
  if (c.blocked.length) {
    pills.push({ tone: "bad", label: d.copyBlocked(c.blocked.length), title: c.blocked.map((b) => `${b.rule_id}: "${b.match}"`).join("\n") });
  } else if (c.copy_blocks > 0) {
    pills.push({ tone: "good", label: d.copyChecked(c.copy_blocks), title: `rules ${c.rules_version}` });
  }
  if (c.missing_disclaimers.length) pills.push({ tone: "warn", label: d.missingDisclaimer, title: c.missing_disclaimers.join(", ") });
  if (c.warnings.length) pills.push({ tone: "warn", label: d.warnings(c.warnings.length), title: c.warnings.map((w) => `${w.rule_id}: "${w.match}"`).join("\n") });
  return pills;
}

export function GuardrailBar({ guardrails, lang }: { guardrails: Guardrails; lang: Language }) {
  const pills = guardrailPills(guardrails, lang);
  if (!pills.length) return null;
  return (
    <div className="guardrails" role="list">
      {pills.map((p, i) => (
        <span key={i} role="listitem" className={`pill ${p.tone}`} title={p.title}>
          {p.label}
        </span>
      ))}
    </div>
  );
}
