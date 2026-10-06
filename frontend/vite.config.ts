import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const backend = env.VITE_BACKEND || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      // Same-origin in dev and prod: the UI always talks to relative /api and /ws.
      proxy: {
        "/api": backend,
        "/healthz": backend,
        "/ws": { target: backend, ws: true },
      },
    },
    test: { environment: "node", include: ["src/**/*.test.ts"] },
  };
});
