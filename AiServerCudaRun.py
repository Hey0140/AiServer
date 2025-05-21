from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import Optional
import shutil
import os
import uuid
import httpx
import subprocess
import json
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

MAIN_SERVER_IP_URL = os.getenv("MAIN_SERVER_IP_URL")
UPLOAD_FOLDER = "uploads/"
OUTPUT_FOLDER = "outputs/"
TARGET_VIDEO_PATHS = [
    "C:/AiServer/AiServer/target1.mp4", "C:/AiServer/AiServer/target2.mp4",
    "C:/AiServer/AiServer/target2.mp4", "C:/AiServer/AiServer/target3.mp4",
    "C:/AiServer/AiServer/target3.mp4", "C:/AiServer/AiServer/target4.mp4",
    "C:/AiServer/AiServer/target4.mp4", "C:/AiServer/AiServer/target5.mp4"
]  # 최대 4개 처리 가능

# 전역 경로 저장
SOURCE_IMAGE_PATH = None

if MAIN_SERVER_IP_URL is None:
    raise ValueError("MAIN_SERVER_IP_URL 환경변수가 설정되지 않았습니다.")

MAIN_SERVER_UPLOAD_URL = f"http://{MAIN_SERVER_IP_URL}:8000/upload_result/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    expected_key = os.getenv("API_KEY")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized")

def create_job_from_basic(source_path, target_path, output_path, settings):
    project_root = os.path.dirname(os.path.abspath(__file__))
    basic_job_path = os.path.join(project_root, "facefusion", ".jobs", "queued", "basic4.json")
    drafted_folder = os.path.join(project_root, "facefusion", ".jobs", "drafted")
    os.makedirs(drafted_folder, exist_ok=True)

    job_id = uuid.uuid4().hex[:5]
    job_filename = f"{job_id}.json"
    new_job_path = os.path.join(drafted_folder, job_filename)

    shutil.copyfile(basic_job_path, new_job_path)

    with open(new_job_path, "r") as f:
        job_data = json.load(f)

    args = job_data["steps"][0]["args"]
    args["source_paths"][0] = os.path.abspath(source_path)
    args["target_path"] = target_path
    args["output_path"] = output_path
    args["processors"] = ["face_swapper", "face_enhancer"]
    args["face_swapper_model"] = settings["face_swapper_model"]
    args["face_enhancer_model"] = settings["face_enhancer_model"]
    args["face_detector_model"] = settings["face_detector_model"]

    with open(new_job_path, "w") as f:
        json.dump(job_data, f, indent=4)

    print(f"[+] Job file created at {new_job_path}")
    return job_id, os.path.abspath(output_path)

def run_facefusion_with_job(job_id, execution_settings=None):
    if execution_settings is None:
        execution_settings = {}

    subprocess.run(["python", "facefusion.py", "job-submit", job_id], cwd="facefusion", check=True)

    command = ["python", "facefusion.py", "job-run", job_id]
    if "execution-providers" in execution_settings:
        command += ["--execution-providers", execution_settings["execution-providers"]]

    subprocess.run(command, cwd="facefusion", check=True)
    print(f"[+] Job {job_id} executed with settings: {execution_settings}")

async def send_output_to_main_server(file_path):
    async with httpx.AsyncClient(timeout=200.0) as client:
        with open(file_path, "rb") as f:
            files = {'file': (os.path.basename(file_path), f, 'video/mp4')}
            headers = {"X-API-KEY": os.getenv("API_KEY")}
            await client.post(MAIN_SERVER_UPLOAD_URL, files=files, headers=headers)
    print(f"[+] Sent result to main server")
    print(f"[+] result path is {file_path}")

@app.post("/run_ai/")
async def run_ai(file: Optional[UploadFile] = File(None),
                 index: int = Form(...),
                 _: None = Depends(verify_api_key)):
    global SOURCE_IMAGE_PATH

    if index == -1:
        print("[==========] DONE signal recieved. Going idle")
        return {"status" : "idle"}

    if file is None:
        raise HTTPException(status_code=400, detail="File is required for processing.")

    # 최초 업로드
    if SOURCE_IMAGE_PATH is None:
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        SOURCE_IMAGE_PATH = save_path
        print(f"[+] Source image saved at {SOURCE_IMAGE_PATH}")
    print(f"[+] index number is {index}")

    if index >= len(TARGET_VIDEO_PATHS):
        raise HTTPException(status_code=400, detail="Invalid index.")

    target_path = TARGET_VIDEO_PATHS[index]
    output_path = f"C:/AiServer/AiServer/outputs/output_{index}.mp4"

    settings = {
        "face_swapper_model": "inswapper_128_fp16",
        "face_enhancer_model": "gfpgan_1.4",
        "face_detector_model": "scrfd"
    }
    execution_settings = {
        "execution-providers": "cuda"
    }

    job_id, _ = create_job_from_basic(
        source_path=SOURCE_IMAGE_PATH,
        target_path=target_path,
        output_path=output_path,
        settings=settings
    )
    run_facefusion_with_job(job_id, execution_settings)
    await send_output_to_main_server(output_path)

    return {"status": f"completed index {index}"}
