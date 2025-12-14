import WebSocket from "ws";
import { spawn } from "child_process";
import server from "./app";
import * as dotenv from "dotenv";
dotenv.config();
server();

const URL = process.env.WS_URL_TARGET;
if (!URL) throw new Error("Environment WS_URL_TARGET belum di-set!");

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
  '-hls_list_size', '10',                                  // jangan terlalu kecil
  '-hls_delete_threshold', '5',                            // hapus segmen lama perlahan
  '-hls_flags', 'delete_segments+independent_segments',    // jangan pakai append_list untuk VLC
  '-master_pl_name', 'output.m3u8',                        // jaga playlist tetap stabil

  process.cwd() + '/output/output.m3u8'
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
  console.log("WebSocket closed");
  ffmpeg.stdin.end();
});

// Logging error ffmpeg
ffmpeg.stderr.on("data", (d) => console.log("FFMPEG:", d.toString()));
ffmpeg.on("close", () => console.log("FFmpeg stopped"));
