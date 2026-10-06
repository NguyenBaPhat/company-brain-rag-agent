// Citations are chunk ids in square brackets: [product-glow-serum#how-to-use]
export const CITATION_RE = /\[([a-z0-9][a-z0-9_-]*#[a-z0-9][a-z0-9_-]*)\]/g;
const UNVERIFIED = "[unverified]";

/** Ordered, de-duplicated list of chunk ids cited in a text (order = first appearance). */
export function extractCitations(text: string): string[] {
  const seen = new Set<string>();
  for (const m of text.matchAll(CITATION_RE)) seen.add(m[1]);
  return [...seen];
}

/**
 * Rewrite citations into markdown links with a `cite:` scheme so the Markdown renderer can turn
 * them into numbered chips: `[1](cite:product-glow-serum%23how-to-use)`.
 * `[unverified]` markers (inserted by the backend guardrail) become `unverified:` links.
 */
export function linkifyCitations(text: string): string {
  const order = extractCitations(text);
  return text
    .replace(CITATION_RE, (_, id: string) => `[${order.indexOf(id) + 1}](cite:${encodeURIComponent(id)})`)
    .split(UNVERIFIED)
    .join("[?](unverified:claim)");
}

export function decodeCitationHref(href: string): string | null {
  return href.startsWith("cite:") ? decodeURIComponent(href.slice(5)) : null;
}

/** Citations of the most recent assistant answer. Empty while that answer is still streaming or if it has none. */
export function latestCitations(messages: { role: string; guardrails?: { citations: { cited: string[] } } | null }[]): string[] {
  const last = messages.at(-1);
  return last?.role === "assistant" ? (last.guardrails?.citations.cited ?? []) : [];
}
