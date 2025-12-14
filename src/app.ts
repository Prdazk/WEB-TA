import express from "express";
import path from "path";
export default function server() {
const app = express();

// Folder output HLS
const hlsPath = path.join(process.cwd(), "output");

// MIME fix

app.use((req, res, next) => {
  if (req.url.endsWith(".m3u8"))
    res.setHeader("Content-Type", "application/vnd.apple.mpegurl");

  if (req.url.endsWith(".ts"))
    res.setHeader("Content-Type", "video/mp2t");

  next();
});

// Serve folder output
app.use("/hls", express.static(hlsPath));

app.listen(3000, () => {
  console.log(`Server running: http://localhost:3000/hls/output.m3u8`);
});
}

