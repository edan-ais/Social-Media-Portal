from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, torch, shutil, cv2, numpy as np
from datetime import datetime
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from PIL import Image
import clip

app = FastAPI()

# --- Auto-create folders on startup ---
for d in ["uploads", "archive", "output"]:
    os.makedirs(d, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

INSTAGRAM_WIDTH, INSTAGRAM_HEIGHT = 1080, 1920
MAX_FRAMES_PER_VIDEO, MIN_CLIP_DURATION = 5, 0.5

def get_image_embedding(frame):
    if frame.dtype == 'float64' or frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image_input).squeeze(0)

def get_clip_embedding(path, n_frames=MAX_FRAMES_PER_VIDEO):
    try:
        clip_obj = VideoFileClip(path)
        if clip_obj.duration < MIN_CLIP_DURATION:
            return None
        times = [clip_obj.duration * i/(n_frames+1) for i in range(1, n_frames+1)]
        embeds = [get_image_embedding(clip_obj.get_frame(t)) for t in times]
        return torch.mean(torch.stack(embeds), dim=0)
    except:
        return None

@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

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
        return {"error": "No files to process."}

    clip_data = []
    video_exts = (".mp4",".mov",".avi",".mkv")
    image_exts = (".jpg",".jpeg",".png",".webp")

    for f in files:
        if f.lower().endswith(video_exts):
            emb = get_clip_embedding(f)
            if emb is not None:
                clip_data.append({"path": f, "embedding": emb, "type":"video"})
        elif f.lower().endswith(image_exts):
            clip_data.append({"path": f, "type":"image"})

    vids = [c for c in clip_data if c["type"]=="video"]
    ordered = []
    if vids:
        current = vids.pop(0)
        ordered.append(current)
        while vids:
            sims = [torch.cosine_similarity(current["embedding"], r["embedding"], dim=0) for r in vids]
            next_idx = sims.index(max(sims))
            current = vids.pop(next_idx)
            ordered.append(current)

    clips = []
    for c in ordered:
        v = VideoFileClip(c["path"]).resize(height=INSTAGRAM_HEIGHT)
        x_center = v.w // 2
        if v.w >= INSTAGRAM_WIDTH:
            v = v.crop(x_center=x_center, width=INSTAGRAM_WIDTH)
        clips.append(v)

    for c in clip_data:
        if c["type"]=="image":
            img = ImageClip(c["path"]).set_duration(3).resize(height=INSTAGRAM_HEIGHT)
            x_center = img.w // 2
            if img.w >= INSTAGRAM_WIDTH:
                img = img.crop(x_center=x_center, width=INSTAGRAM_WIDTH)
            clips.append(img)

    if not clips:
        return {"error": "No usable clips."}

    output_path = os.path.join("output", f"montage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_path, fps=30)

    for f in files:
        shutil.move(f, os.path.join("archive", os.path.basename(f)))

    return {"status": "done", "output": output_path}

@app.get("/output/{filename}")
async def get_output(filename: str):
    path = os.path.join("output", filename)
    if not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(path, media_type="video/mp4")
