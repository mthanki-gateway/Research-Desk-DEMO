import PageSkeleton from "../page-skeleton";

/**
 * Shown instantly while /chat or /chat/[id] loads.
 *
 * Note this does NOT replace the in-page cache in `chat/[id]/page.tsx`: that
 * cache is what stops a *already-visited* conversation flashing a skeleton.
 * This fallback covers the genuinely-cold navigation, where there is nothing
 * cached to show.
 */
export default function Loading() {
  return <PageSkeleton rows={2} />;
}
