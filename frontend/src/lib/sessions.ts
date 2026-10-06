import type { SessionSummary } from "./api";

export type Bucket = "today" | "yesterday" | "week" | "older";
export const BUCKET_ORDER: Bucket[] = ["today", "yesterday", "week", "older"];

const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();

export function bucketOf(updatedAtSec: number, now: Date = new Date()): Bucket {
  const day = 86_400_000;
  const diff = startOfDay(now) - startOfDay(new Date(updatedAtSec * 1000));
  if (diff <= 0) return "today";
  if (diff <= day) return "yesterday";
  if (diff <= 7 * day) return "week";
  return "older";
}

/** Group newest-first sessions into Today / Yesterday / Previous 7 days / Older (empty groups omitted). */
export function groupSessions(sessions: SessionSummary[], now: Date = new Date()): { bucket: Bucket; items: SessionSummary[] }[] {
  const groups = new Map<Bucket, SessionSummary[]>();
  for (const s of sessions) {
    const b = bucketOf(s.updated_at, now);
    groups.set(b, [...(groups.get(b) ?? []), s]);
  }
  return BUCKET_ORDER.filter((b) => groups.has(b)).map((bucket) => ({ bucket, items: groups.get(bucket)! }));
}
