import os, shutil, subprocess, uuid
from datetime import datetime

import torch
import numpy as np
import cv2
from PIL import Image
import imageio_ffmpeg
import imageio.v3 as iio

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse

import clip

# --- Folders ---
UPLOAD_DIR = "uploads"
ARCHIVE_DIR = "archive"
OUTPUT_DIR = "output"
for d in (UPLOAD_DIR, ARCHIVE_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# --- Model / settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

INSTAGRAM_W, INSTAGRAM_H = 1080, 1920
MAX_FRAMES_PER_VIDEO = 5
MIN_CLIP_DURATION = 0.5  # seconds

ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI()


def get_image_embedding_from_ndarray(frame_bgr: np.ndarray) -> torch.Tensor:
    # ensure dtype uint8 [0..255]
    if frame_bgr.dtype != np.uint8:
        frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image_input).squeeze(0)


def sample_video_embedding(path: str, n_frames: int = MAX_FRAMES_PER_VIDEO) -> torch.Tensor | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / max(fps, 1e-6)
    if duration < MIN_CLIP_DURATION or frames <= 0:
        cap.release()
        return None

    times = [duration * i / (n_frames + 1) for i in range(1, n_frames + 1)]
    embeds = []
    for t in times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        embeds.append(get_image_embedding_from_ndarray(frame))
    cap.release()
    if not embeds:
        return None
    return torch.mean(torch.stack(embeds), dim=0)


def normalize_video_to_segment(src: str, dst: str):
    """
    Scale to height 1920, then center-crop width 1080.
    Re-encode as H.264/AAC MP4 for concat compatibility.
    """
    # Two-pass filter: scale then crop centered to 1080x1920.
    vf = (
        f"scale=-2:{INSTAGRAM_H}:force_original_aspect_ratio=decrease,"
        f"pad=iw:ih:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"crop={INSTAGRAM_W}:{INSTAGRAM_H}"
    )
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", src,
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        dst,
    ]
    subprocess.run(cmd, check=True)


def normalize_image_to_segment(src: str, dst: str, duration_sec: int = 3):
    """
    Turn image into 3s 1080x1920 mp4 with letterbox/crop as needed.
    """
    vf = (
        f"scale=-2:{INSTAGRAM_H}:force_original_aspect_ratio=decrease,"
        f"pad={INSTAGRAM_W}:{INSTAGRAM_H}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loop", "1",
        "-t", str(duration_sec),
        "-i", src,
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        "-an",
        dst,
    ]
    subprocess.run(cmd, check=True)


def concat_segments(segment_paths: list[str], out_path: str):
    # use concat demuxer for robustness
    listfile = f"/tmp/concat_{uuid.uuid4().hex}.txt"
    with open(listfile, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", listfile,
        "-c", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(listfile)


@app.get("/")
async def root():
    # minimal UI
    html = """
    <!DOCTYPE html>
    <html><body style="font-family:sans-serif;text-align:center;margin-top:40px">
      <h2>Upload & Create Montage</h2>
      <input type="file" id="file" multiple />
      <button onclick="upload()">Upload</button>
      <button onclick="process()">Create Montage</button>
      <p id="status"></p>
      <video id="video" width="360" controls style="display:none;"></video>
    <script>
    async function upload(){
      const files = document.getElementById('file').files;
      for (const f of files){
        const fd = new FormData(); fd.append('file', f);
        await fetch('/upload', {method:'POST', body: fd});
      }
      document.getElementById('status').innerText = 'Uploaded.';
    }
    async function process(){
      document.getElementById('status').innerText = 'Processing...';
      const res = await fetch('/process', {method:'POST'});
      const data = await res.json();
      if(data.output){
        document.getElementById('status').innerText = 'Done!';
        const v = document.getElementById('video');
        v.src = '/' + data.output; v.style.display = 'block';
      } else {
        document.getElementById('status').innerText = data.error || 'Error';
      }
    }
    </script>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"status": "uploaded", "filename": file.filename}


@app.post("/process")
async def process_media():
    files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)]
    if not files:
        return {"error": "No files to process."}

    video_exts = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
    image_exts = (".jpg", ".jpeg", ".png", ".webp")

    # Collect embeddings (videos) and pass images as-is
    clip_data = []
    for p in files:
        low = p.lower()
        if low.endswith(video_exts):
            emb = sample_video_embedding(p)
            if emb is not None:
                clip_data.append({"path": p, "embedding": emb, "type": "video"})
        elif low.endswith(image_exts):
            clip_data.append({"path": p, "type": "image"})

    # Order videos by cosine similarity chain
    vids = [c for c in clip_data if c["type"] == "video"]
    ordered = []
    if vids:
        remaining = vids[:]
        current = remaining.pop(0)
        ordered.append(current)
        while remaining:
            sims = [torch.cosine_similarity(current["embedding"], r["embedding"], dim=0).item() for r in remaining]
            next_idx = int(np.argmax(sims))
            current = remaining.pop(next_idx)
            ordered.append(current)

    # Build normalized segments
    tmp_dir = f"/tmp/segments_{uuid.uuid4().hex}"
    os.makedirs(tmp_dir, exist_ok=True)
    segments = []

    try:
        # videos first (ordered), then images
        for c in ordered:
            dst = os.path.join(tmp_dir, f"seg_{len(segments):04d}.mp4")
            normalize_video_to_segment(c["path"], dst)
            segments.append(dst)

        for c in clip_data:
            if c["type"] == "image":
                dst = os.path.join(tmp_dir, f"seg_{len(segments):04d}.mp4")
                normalize_image_to_segment(c["path"], dst, duration_sec=3)
                segments.append(dst)

        if not segments:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {"error": "No usable clips."}

        out_name = f"montage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        concat_segments(segments, out_path)

        # archive used inputs
        for p in files:
            shutil.move(p, os.path.join(ARCHIVE_DIR, os.path.basename(p)))

        return {"status": "done", "output": os.path.join(OUTPUT_DIR, out_name)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/output/{filename}")
async def get_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(path, media_type="video/mp4")
