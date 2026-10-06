import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { decodeCitationHref, linkifyCitations } from "../lib/citations";
import type { Source } from "../types";

interface Props {
  text: string;
  sources: Record<string, Source>;
  activeCitation: string | null;
  onCite: (chunkId: string) => void;
}

const SAFE_PROTOCOLS = /^(https?:|mailto:|cite:|unverified:|\/api\/)/i;
// react-markdown strips unknown URL schemes by default; allow our own `cite:` / `unverified:` ones.
const urlTransform = (url: string) => (SAFE_PROTOCOLS.test(url) ? url : "");

export function Markdown({ text, sources, activeCitation, onCite }: Props) {
  const components: Components = {
    a({ href = "", children }) {
      const chunkId = decodeCitationHref(href);
      if (chunkId) {
        const deprecated = sources[chunkId]?.status === "deprecated";
        const cls = ["cite", chunkId === activeCitation ? "active" : "", deprecated ? "deprecated" : ""].join(" ");
        return (
          <button type="button" className={cls} title={chunkId} onClick={() => onCite(chunkId)}>
            {children}
          </button>
        );
      }
      if (href.startsWith("unverified:")) {
        return <span className="cite unverified" title="Citation removed: not retrieved from the knowledge base">?</span>;
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      );
    },
  };
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components} urlTransform={urlTransform}>
        {linkifyCitations(text)}
      </ReactMarkdown>
    </div>
  );
}
