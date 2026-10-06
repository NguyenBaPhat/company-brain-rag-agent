import { useEffect, useMemo, useRef, useState } from "react";
import { fetchChunk, fetchDocuments, type KbDocument } from "../lib/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { t } from "../i18n";
import type { Language, Source } from "../types";

interface Props {
  lang: Language;
  tab: "sources" | "kb";
  onTab: (tab: "sources" | "kb") => void;
  citationIds: string[];
  activeCitation: string | null;
  sources: Record<string, Source>;
}

function SourceCard({ id, lang, source, active, index }: { id: string; lang: Language; source: Source | undefined | null; active: boolean; index: number }) {
  const d = t(lang);
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [active]);
  return (
    <article ref={ref} className={`source ${active ? "active" : ""}`}>
      <header>
        <span className="cite static">{index}</span>
        <div>
          <div className="source-doc">{source?.document ?? id.split("#")[0]}</div>
          <div className="source-section">{source?.section ?? id.split("#")[1]}</div>
        </div>
        {source?.status && <span className={`badge ${source.status}`}>{source.status === "deprecated" ? d.deprecated : d.active}</span>}
      </header>
      {source ? (
        <div className="source-text">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{source.text}</ReactMarkdown>
        </div>
      ) : (
        <p>{d.loading}</p>
      )}
      <footer>
        <code>{id}</code>
        {source?.owner && <span>{d.owner}: {source.owner}</span>}
        {typeof source?.relevance === "number" && <span>{d.relevance}: {Math.round(source.relevance * 100)}%</span>}
      </footer>
    </article>
  );
}

function SourcesTab({ lang, citationIds, activeCitation, sources }: Omit<Props, "tab" | "onTab">) {
  const d = t(lang);
  const [fetched, setFetched] = useState<Record<string, Source | null>>({});
  const requested = useRef(new Set<string>());

  // Citations restored from history have no retrieval payload in memory: fetch them on demand.
  useEffect(() => {
    for (const id of citationIds) {
      if (sources[id] || requested.current.has(id)) continue;
      requested.current.add(id);
      fetchChunk(id)
        .then((s) => setFetched((f) => ({ ...f, [id]: s })))
        .catch(() => setFetched((f) => ({ ...f, [id]: null })));
    }
  }, [citationIds, sources]);

  if (!citationIds.length) return <p className="empty-note">{d.noSources}</p>;
  return (
    <div className="source-list">
      {citationIds.map((id, i) => (
        <SourceCard key={id} id={id} lang={lang} index={i + 1} active={id === activeCitation} source={sources[id] ?? fetched[id]} />
      ))}
    </div>
  );
}

function KnowledgeTab({ lang }: { lang: Language }) {
  const d = t(lang);
  const [docs, setDocs] = useState<KbDocument[] | null>(null);
  useEffect(() => {
    fetchDocuments().then(setDocs).catch(() => setDocs([]));
  }, []);
  const groups = useMemo(() => {
    const by: Record<string, KbDocument[]> = {};
    for (const x of docs ?? []) (by[x.doc_type] ??= []).push(x);
    return Object.entries(by);
  }, [docs]);
  if (!docs) return <p className="empty-note">{d.loading}</p>;
  return (
    <div className="kb-list">
      {groups.map(([type, items]) => (
        <section key={type}>
          <h4>{d.docTypes[type] ?? type}</h4>
          <ul>
            {items.map((x) => (
              <li key={x.doc_id} className={x.status}>
                <span className="kb-title">{x.title}</span>
                <span className="kb-meta">
                  {x.chunks} {d.chunksLabel} · {x.updated}
                  {x.status === "deprecated" && <span className="badge deprecated">{d.deprecated}</span>}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

export function SidePanel(props: Props) {
  const d = t(props.lang);
  return (
    <aside className="side">
      <div className="tabs" role="tablist">
        {(["sources", "kb"] as const).map((tab) => (
          <button key={tab} role="tab" aria-selected={props.tab === tab} className={props.tab === tab ? "on" : ""} onClick={() => props.onTab(tab)}>
            {tab === "sources" ? `${d.sources}${props.citationIds.length ? ` (${props.citationIds.length})` : ""}` : d.knowledgeBase}
          </button>
        ))}
      </div>
      <div className="side-body">{props.tab === "sources" ? <SourcesTab {...props} /> : <KnowledgeTab lang={props.lang} />}</div>
    </aside>
  );
}
