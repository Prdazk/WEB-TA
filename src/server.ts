import WebSocket from "ws";
import { spawn } from "child_process";
import server from "./app";
import * as dotenv from "dotenv";
dotenv.config();

server();

// Ambil 2 URL dari .env
const URLS = [
  process.env.WS_URL_TARGET1,
  process.env.WS_URL_TARGET2
];

URLS.forEach((URL, index) => {
  if (!URL) throw new Error(`Environment WS_URL_TARGET${index + 1} belum di-set!`);

  const ws = new WebSocket(URL);

  const ffmpeg = spawn('ffmpeg', [
    '-i', 'pipe:0',
    '-fflags', 'nobuffer',
    '-flags', 'low_delay',
    '-c:v', 'copy',
    '-preset', 'ultrafast',

    // HLS stabil untuk live 24 jam
    '-f', 'hls',
    '-hls_time', '1',
    '-hls_list_size', '10',
    '-hls_delete_threshold', '5',
    '-hls_flags', 'delete_segments+independent_segments',
    '-master_pl_name', `output${index + 1}.m3u8`,   // Output berbeda per CCTV

    process.cwd() + `/output/output${index + 1}.m3u8`
  ]);

  ws.on("message", (data: Buffer) => {
    const ok = ffmpeg.stdin.write(data);
    if (!ok) {
      ws.pause();
      ffmpeg.stdin.once("drain", () => ws.resume());
    }
  });

  // heartbeat supaya websocket tidak close
  setInterval(() => {
    if (ws.readyState === ws.OPEN) ws.ping();
  }, 10000);

  ws.on("close", () => {
    console.log(`WebSocket ${index + 1} closed`);
    ffmpeg.stdin.end();
  });

  // Logging error ffmpeg
  ffmpeg.stderr.on("data", (d) => console.log(`FFMPEG ${index + 1}:`, d.toString()));
  ffmpeg.on("close", () => console.log(`FFmpeg ${index + 1} stopped`));
});
