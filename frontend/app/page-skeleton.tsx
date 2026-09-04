/**
 * Shared navigation placeholder.
 *
 * Next's App Router renders the nearest `loading.tsx` as a Suspense fallback
 * the INSTANT a navigation starts. Without one, the router keeps the old page
 * on screen until the new route's data resolves — so a slow page looks like a
 * dead click rather than a loading one. That is the "lag" when moving between
 * Chat, Library and Lab: nothing was broken, there was simply no feedback.
 *
 * Deliberately a skeleton rather than a spinner. M3 guidance prefers a
 * placeholder that mirrors the incoming layout, because it communicates *what*
 * is arriving, and the eye has somewhere to rest. A centred spinner tells you
 * only that something is happening.
 */
export default function PageSkeleton({
  title = true,
  rows = 3,
}: {
  title?: boolean;
  rows?: number;
}) {
  return (
    <div className="mx-auto max-w-3xl px-6 py-6" aria-busy="true">
      {/* Announced once, politely — a screen reader should hear "Loading"
          rather than every skeleton block. */}
      <span className="sr-only" role="status">
        Loading
      </span>

      {title && (
        <div className="mb-6 flex h-16 items-center">
          <div className="md-skeleton h-8 w-56 rounded-[var(--md-shape-sm)]" />
        </div>
      )}

      <div className="space-y-4">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="md-answer p-5">
            <div className="space-y-3">
              <div className="md-skeleton h-4 w-full rounded-[var(--md-shape-xs)]" />
              <div className="md-skeleton h-4 w-11/12 rounded-[var(--md-shape-xs)]" />
              <div className="md-skeleton h-4 w-4/6 rounded-[var(--md-shape-xs)]" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
