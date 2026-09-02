import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `standalone` emits a self-contained server bundle plus .next/standalone,
  // which the prod stage of frontend/Dockerfile copies. It is required for
  // self-hosting and MUST NOT be set on Vercel: Vercel builds via its own
  // Build Output API, and standalone mode relocates the file-trace manifests it
  // then fails to find --
  //   ENOENT: ... open '/vercel/path0/frontend/.next/next-server.js.nft.json'
  // which reads like a corrupt install rather than a config conflict.
  //
  // VERCEL=1 is set by Vercel in every build environment, so this picks the
  // right mode with no manual flag to remember.
  //
  // Neither a local `next build` nor CI can catch this: both succeed with
  // standalone output, because the step that breaks is Vercel's own
  // post-build trace collection. Only a real Vercel deploy exercises it.
  output: process.env.VERCEL ? undefined : "standalone",
};

export default nextConfig;
