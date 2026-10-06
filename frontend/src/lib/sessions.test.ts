import { describe, expect, it } from "vitest";
import type { SessionSummary } from "./api";
import { bucketOf, groupSessions } from "./sessions";

const now = new Date(2026, 9, 6, 15, 0, 0); // 6 Oct 2026, 15:00 local
const at = (daysAgo: number, hour = 9) => new Date(2026, 9, 6 - daysAgo, hour).getTime() / 1000;
const s = (id: string, updated_at: number): SessionSummary => ({ id, title: id, updated_at, language: "en" });

describe("session grouping", () => {
  it("buckets by calendar day, not by 24h windows", () => {
    expect(bucketOf(at(0, 1), now)).toBe("today");
    expect(bucketOf(at(1, 23), now)).toBe("yesterday");
    expect(bucketOf(at(3), now)).toBe("week");
    expect(bucketOf(at(7), now)).toBe("week");
    expect(bucketOf(at(8), now)).toBe("older");
  });

  it("keeps order inside a group and omits empty groups", () => {
    const groups = groupSessions([s("a", at(0)), s("b", at(0, 8)), s("c", at(30))], now);
    expect(groups.map((g) => g.bucket)).toEqual(["today", "older"]);
    expect(groups[0].items.map((x) => x.id)).toEqual(["a", "b"]);
  });

  it("handles an empty list", () => {
    expect(groupSessions([], now)).toEqual([]);
  });
});
