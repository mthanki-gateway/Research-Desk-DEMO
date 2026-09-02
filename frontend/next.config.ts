import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // needed by the prod Docker stage; harmless on Vercel
  output: "standalone",
};

export default nextConfig;
