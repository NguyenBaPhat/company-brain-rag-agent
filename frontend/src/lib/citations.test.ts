import { describe, expect, it } from "vitest";
import { decodeCitationHref, extractCitations, latestCitations, linkifyCitations } from "./citations";

describe("citations", () => {
  const text = "Apply 2 pumps [product-glow-serum#how-to-use]. Price $38 [product-glow-serum#overview] again [product-glow-serum#how-to-use].";

  it("extracts unique ids in order of first appearance", () => {
    expect(extractCitations(text)).toEqual(["product-glow-serum#how-to-use", "product-glow-serum#overview"]);
  });

  it("numbers repeated citations consistently", () => {
    const out = linkifyCitations(text);
    expect(out.match(/\[1\]\(cite:/g)).toHaveLength(2);
    expect(out).toContain("[2](cite:product-glow-serum%23overview)");
  });

  it("round-trips the encoded id", () => {
    expect(decodeCitationHref("cite:product-glow-serum%23how-to-use")).toBe("product-glow-serum#how-to-use");
    expect(decodeCitationHref("https://example.com")).toBeNull();
  });

  it("marks unverified citations", () => {
    expect(linkifyCitations("Claim [unverified].")).toBe("Claim [?](unverified:claim).");
  });

  it("ignores things that are not citations", () => {
    expect(extractCitations("array[0] and [link](x) and [Not#Valid]")).toEqual([]);
  });
});

describe("latestCitations (what the Sources panel shows)", () => {
  const answered = { role: "assistant", guardrails: { citations: { cited: ["a#b", "c#d"] } } };

  it("shows the citations of the latest finished answer", () => {
    expect(latestCitations([{ role: "user" }, answered])).toEqual(["a#b", "c#d"]);
  });

  it("is empty as soon as a new question is asked (new streaming answer has no guardrails yet)", () => {
    expect(latestCitations([{ role: "user" }, answered, { role: "user" }, { role: "assistant", guardrails: undefined }])).toEqual([]);
  });

  it("is empty for a new/empty conversation and for an answer without citations", () => {
    expect(latestCitations([])).toEqual([]);
    expect(latestCitations([{ role: "assistant", guardrails: { citations: { cited: [] } } }])).toEqual([]);
  });
});
