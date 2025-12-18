import express from "express";
import path from "path";

export default function server(port = 3000) {
  const app = express();

  const rootDir = process.cwd();
  const hlsPath = path.join(rootDir, "output");
  const publicPath = path.join(rootDir, "public");

  /* ===== CORS (WAJIB) ===== */
  app.use((req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    next();
  });

  /* ===== MIME FIX HLS ===== */
  app.use((req, res, next) => {
    if (req.path.endsWith(".m3u8")) {
      res.setHeader("Content-Type", "application/vnd.apple.mpegurl");
      res.setHeader("Cache-Control", "no-cache");
    }

    if (req.path.endsWith(".ts")) {
      res.setHeader("Content-Type", "video/mp2t");
      res.setHeader("Cache-Control", "no-cache");
    }

    next();
  });

  /* ===== STATIC HLS ===== */
  app.use(
    "/hls",
    express.static(hlsPath, {
      index: false,
      fallthrough: true
    })
  );

  /* ===== STATIC WEB ===== */
  app.use(
    "/",
    express.static(publicPath, {
      index: "index.html"
    })
  );

  /* ===== HEALTH CHECK ===== */
  app.get("/api/health", (_, res) => {
    res.json({
      status: "ok",
      hls: "/hls/{stream-name}/output.m3u8"
    });
  });

  app.listen(port, () => {
    console.log(`🚀 Server running on http://localhost:${port}`);
    console.log(`🎥 HLS example: http://localhost:${port}/hls/helmet_pnm/output.m3u8`);
  });
}
