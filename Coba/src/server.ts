import WebSocket from "ws";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import * as dotenv from "dotenv";
import server from "./app";

dotenv.config();
server();

/* ===================== UTIL ===================== */
function safeNameFromURL(url: any) {
  return url
    .replace(/^wss?:\/\//, "")
    .replace(/[^a-zA-Z0-9]/g, "_")
    .toLowerCase();
}

function ensureDir(dir: any) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

/* ===================== CORE ===================== */
function startStream(wsURL: any) {
  const name = safeNameFromURL(wsURL);
  const outputDir = path.join(process.cwd(), "output", name);
  ensureDir(outputDir);

  console.log(`[STREAM] Start ${name}`);

  const ws = new WebSocket(wsURL);

  const ffmpeg = spawn("ffmpeg", [
    "-i", "pipe:0",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-c:v", "copy",
    "-preset", "ultrafast",
    "-f", "hls",
    "-hls_time", "1",
    "-hls_list_size", "10",
    "-hls_delete_threshold", "5",
    "-hls_flags", "delete_segments+independent_segments",
    "-master_pl_name", "output.m3u8",
    path.join(outputDir, "output.m3u8")
  ]);

  /* ===== WebSocket → FFmpeg ===== */
  ws.on("message", (data) => {
    if (!ffmpeg.stdin.write(data)) {
      ws.pause();
      ffmpeg.stdin.once("drain", () => ws.resume());
    }
  });

  ws.on("close", () => {
    console.log(`[STREAM] WS closed: ${name}`);
    ffmpeg.stdin.end();
  });

  ws.on("error", (e) => {
    console.error(`[WS ERROR] ${name}`, e.message);
  });

  /* ===== Keep Alive ===== */
  const ping = setInterval(() => {
    if (ws.readyState === ws.OPEN) ws.ping();
  }, 10000);

  /* ===== FFmpeg Log ===== */
  ffmpeg.stderr.on("data", (d) => {
    console.log(`[FFMPEG ${name}] ${d.toString()}`);
  });

  ffmpeg.on("close", () => {
    clearInterval(ping);
    console.log(`[FFMPEG] Closed ${name}`);
  });
}

/* ===================== MULTI STREAM ===================== */
const urls = process.env.WS_URL_TARGET?.split(",").map(v => v.trim());

if (!urls || !urls.length) {
  throw new Error("WS_URL_TARGET belum di-set!");
}

urls.forEach(startStream);
