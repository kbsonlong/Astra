import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const runtimeEnv = (globalThis as { process?: { env?: Record<string, string | undefined> } })
    .process?.env;
  const backendPort = runtimeEnv?.BACKEND_PORT || env.BACKEND_PORT || "8001";

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
          target: `http://127.0.0.1:${backendPort}`,
          ws: true,
        },
        "/ws": {
          target: `ws://127.0.0.1:${backendPort}`,
          ws: true,
        },
      },
    },
  };
});
