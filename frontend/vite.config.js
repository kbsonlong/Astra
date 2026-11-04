import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(function (_a) {
    var _b;
    var mode = _a.mode;
    var env = loadEnv(mode, ".", "");
    var runtimeEnv = (_b = globalThis
        .process) === null || _b === void 0 ? void 0 : _b.env;
    var backendPort = (runtimeEnv === null || runtimeEnv === void 0 ? void 0 : runtimeEnv.BACKEND_PORT) || env.BACKEND_PORT || "8001";
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
                    target: "http://127.0.0.1:".concat(backendPort),
                    ws: true,
                },
                "/ws": {
                    target: "ws://127.0.0.1:".concat(backendPort),
                    ws: true,
                },
            },
        },
    };
});
