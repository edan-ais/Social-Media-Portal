import os, subprocess, shutil, uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import torch
import numpy as np
import clip
import cv2
from PIL import Image
import imageio_ffmpeg

# --- Folder setup ---
for folder in ["uploads", "archive", "output"]:
    os.makedirs(folder, exist_ok=True)

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

INSTAGRAM_W, INSTAGRAM_H = 1080, 1920
MAX_FRAMES_PER_VIDEO, MIN_CLIP_DURATION = 5, 0.5
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()


def get_image_embedding(frame_bgr):
    """Convert BGR frame to CLIP embedding."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image_input).squeeze(0)


def get_clip_embedding(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = frames / fps if fps > 0 else 0
    if dur < MIN_CLIP_DURATION:
        cap.release()
        return None
    times = [int(frames * i / (MAX_FRAMES_PER_VIDEO + 1)) for i in range(1, MAX_FRAMES_PER_VIDEO + 1)]
    embeds = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        if ok:
            embeds.append(get_image_embedding(frame))
    cap.release()
    if not embeds:
        return None
    return torch.mean(torch.stack(embeds), dim=0)


def ffmpeg_normalize_video(src, dst):
    vf = f"scale=-2:{INSTAGRAM_H}:force_original_aspect_ratio=decrease,pad={INSTAGRAM_W}:{INSTAGRAM_H}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    subprocess.run([FFMPEG_BIN, "-y", "-i", src, "-vf", vf,
                    "-r", "30", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", dst], check=True)


def ffmpeg_image_to_video(src, dst, duration=3):
    vf = f"scale=-2:{INSTAGRAM_H}:force_original_aspect_ratio=decrease,pad={INSTAGRAM_W}:{INSTAGRAM_H}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    subprocess.run([FFMPEG_BIN, "-y", "-loop", "1", "-t", str(duration), "-i", src, "-vf", vf,
                    "-r", "30", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-an", dst], check=True)


def ffmpeg_concat(files, output):
    concat_list = f"/tmp/concat_{uuid.uuid4().hex}.txt"
    with open(concat_list, "w") as f:
        for fpath in files:
            f.write(f"file '{fpath}'\n")
    subprocess.run([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                    "-c", "copy", output], check=True)
    os.remove(concat_list)


@app.get("/")
async def ui():
    html = """
    <!DOCTYPE html><html><body style="font-family:sans-serif;text-align:center;margin-top:40px">
    <h2>AI Montage Bot</h2>
    <input type="file" id="file" multiple/>
    <button onclick="upload()">Upload</button>
    <button onclick="process()">Create Montage</button>
    <p id="status"></p><video id="video" width="360" controls style="display:none;"></video>
    <script>
    async function upload(){
      const files=document.getElementById('file').files;
      for(const f of files){
        const fd=new FormData();fd.append('file',f);
        await fetch('/upload',{method:'POST',body:fd});
      }
      document.getElementById('status').innerText='Uploaded.';
    }
    async function process(){
      document.getElementById('status').innerText='Processing...';
      const res=await fetch('/process',{method:'POST'});
      const data=await res.json();
      if(data.output){
        document.getElementById('status').innerText='Done!';
        const v=document.getElementById('video');
        v.src='/'+data.output;v.style.display='block';
      }else{document.getElementById('status').innerText=data.error||'Error';}
    }
    </script></body></html>
    """
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = os.path.join("uploads", file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"status": "uploaded", "filename": file.filename}


@app.post("/process")
async def process_videos():
    files = [os.path.join("uploads", f) for f in os.listdir("uploads")]
    if not files:
        return {"error": "No files uploaded."}

    video_exts = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
    image_exts = (".jpg", ".jpeg", ".png", ".webp")
    data = []
    for f in files:
        if f.lower().endswith(video_exts):
            emb = get_clip_embedding(f)
            if emb is not None:
                data.append({"path": f, "embedding": emb, "type": "video"})
        elif f.lower().endswith(image_exts):
            data.append({"path": f, "type": "image"})

    vids = [x for x in data if x["type"] == "video"]
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

    tmp_dir = f"/tmp/segments_{uuid.uuid4().hex}"
    os.makedirs(tmp_dir, exist_ok=True)
    segs = []

    try:
        for c in ordered:
            out = os.path.join(tmp_dir, f"seg_{len(segs):03d}.mp4")
            ffmpeg_normalize_video(c["path"], out)
            segs.append(out)

        for c in data:
            if c["type"] == "image":
                out = os.path.join(tmp_dir, f"seg_{len(segs):03d}.mp4")
                ffmpeg_image_to_video(c["path"], out)
                segs.append(out)

        if not segs:
            return {"error": "No usable clips found."}

        out_name = f"montage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join("output", out_name)
        ffmpeg_concat(segs, out_path)

        for f in files:
            shutil.move(f, os.path.join("archive", os.path.basename(f)))

        return {"status": "done", "output": out_path}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/output/{filename}")
async def get_output(filename: str):
    path = os.path.join("output", filename)
    if not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(path, media_type="video/mp4")
