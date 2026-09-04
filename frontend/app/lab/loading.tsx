import PageSkeleton from "../page-skeleton";

/** Shown instantly while /lab loads. See page-skeleton.tsx. */
export default function Loading() {
  return <PageSkeleton rows={2} />;
}
