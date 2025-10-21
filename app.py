import os, uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import httpx

# --- Folder setup ---
for folder in ["uploads"]:
    os.makedirs(folder, exist_ok=True)

app = FastAPI(title="AI Montage Bot (Lightweight Frontend)")

# --- GitHub Actions Config ---
GITHUB_REPO = os.getenv("GH_REPO", "edan-ais/Social-Media-Portal")
GITHUB_TOKEN = os.getenv("GH_PAT")

@app.get("/")
async def ui():
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>AI Montage Bot</title></head>
    <body style="font-family:sans-serif;text-align:center;margin-top:40px">
      <h2>AI Montage Bot</h2>
      <p>Upload your clips and trigger montage creation (runs on GitHub Actions).</p>
      <input type="file" id="file" multiple/>
      <button onclick="upload()">Upload</button>
      <button onclick="trigger()">Trigger Montage Build</button>
      <p id="status"></p>

      <script>
      async function upload(){
        const files=document.getElementById('file').files;
        if(!files.length){document.getElementById('status').innerText='No files selected.';return;}
        for(const f of files){
          const fd=new FormData();fd.append('file',f);
          await fetch('/upload',{method:'POST',body:fd});
        }
        document.getElementById('status').innerText='Files uploaded to Render.';
      }
      async function trigger(){
        document.getElementById('status').innerText='Triggering GitHub Actions...';
        const res=await fetch('/trigger',{method:'POST'});
        const data=await res.json();
        document.getElementById('status').innerText=JSON.stringify(data);
      }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/health")
async def health():
    """Health check endpoint for Render"""
    return {"status": "ok", "service": "frontend", "mode": "lightweight"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accept and store uploaded files (images/videos)"""
    file_id = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join("uploads", file_id)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"status": "uploaded", "filename": file_id}


@app.post("/trigger")
async def trigger_workflow():
    """Trigger the GitHub Actions workflow remotely"""
    if not GITHUB_TOKEN:
        return JSONResponse({"error": "Missing GH_PAT environment variable"}, status_code=400)

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/montage.yml/dispatches"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"ref": "main"}

    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code in (200, 201, 204):
            return {"status": "success", "message": "GitHub Actions workflow triggered!"}
        else:
            return {
                "status": "error",
                "code": r.status_code,
                "details": r.text,
                "url": url
            }
