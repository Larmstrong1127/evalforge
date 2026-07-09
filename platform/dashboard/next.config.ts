import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enables `.next/standalone/server.js` — a self-contained build with only
  // the traced production dependencies, used by the Docker Compose image.
  output: "standalone",
};

export default nextConfig;
