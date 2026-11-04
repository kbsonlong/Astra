import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy upstreams. Defaults target a single-process `all` backend.
// For a split deployment, point /ws at the realtime process and /api at the
// workbench process via VITE_WS_UPSTREAM / VITE_API_UPSTREAM.
// BACKEND_PORT (or VITE env equivalent) still overrides the default port.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const runtimeEnv = (globalThis as { process?: { env?: Record<string, string | undefined> } })
    .process?.env;
  const backendPort = runtimeEnv?.BACKEND_PORT || env.BACKEND_PORT || "8000";
  const apiUpstream = env.VITE_API_UPSTREAM || `127.0.0.1:${backendPort}`;
  const wsUpstream = env.VITE_WS_UPSTREAM || apiUpstream;
  return {
    plugins: [react()],
    test: {
      environment: "node",
      include: ["src/**/*.test.ts"],
    },
    server: {
      host: "0.0.0.0",
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": {
          target: `http://${apiUpstream}`,
          ws: true,
        },
        "/ws": {
          target: `ws://${wsUpstream}`,
          ws: true,
        },
      },
    },
  };
});
