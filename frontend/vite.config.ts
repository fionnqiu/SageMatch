import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// 重启脚本遇到端口占用时会改用下一个端口，并把实际后端端口放进这个变量。
const apiPort = process.env.SAGEMATCH_API_PORT || "8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${apiPort}`,
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes, _req, res) => {
            // 只有事件流关掉缓冲。普通 JSON 仍按 Vite 默认方式转发。
            const type = String(proxyRes.headers["content-type"] || "");
            if (type.includes("text/event-stream")) res.setHeader("X-Accel-Buffering", "no");
          });
        },
      },
    },
  },
});
