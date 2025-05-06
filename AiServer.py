# AiServer.py
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
import shutil
import os
import uuid
import httpx
import subprocess
import json

app = FastAPI()

load_dotenv()

MAIN_SERVER_IP_URL = os.getenv("MAIN_SERVER_IP_URL")
UPLOAD_FOLDER = "uploads/"
OUTPUT_FOLDER = "outputs/"
#TARGET_VIDEO_PATH = "target.mp4"
TARGET_VIDEO_PATH = "test_target.png"

if MAIN_SERVER_IP_URL is None:
    raise ValueError("MAIN_SERVER_IP_URL 환경변수가 설정되지 않았습니다.")

MAIN_SERVER_UPLOAD_URL = "http://"+MAIN_SERVER_IP_URL+":8000/upload_result/"


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def create_job_from_basic(source_path, target_path, output_path, settings):
    project_root = os.path.dirname(os.path.abspath(__file__))
    basic_job_path = os.path.join(project_root, "facefusion", ".jobs", "queued", "basic.json")

    drafted_folder = os.path.join(project_root, "facefusion", ".jobs", "drafted")
    os.makedirs(drafted_folder, exist_ok=True)

    job_id = uuid.uuid4().hex
    job_filename = f"{job_id}.json"
    new_job_path = os.path.join(drafted_folder, job_filename)

    shutil.copyfile(basic_job_path, new_job_path)

    with open(new_job_path, "r") as f:
        job_data = json.load(f)

    args = job_data["steps"][0]["args"]

    def get_absolute_path(path):
        if not os.path.isabs(path):
            return os.path.abspath(path)
        return path

    args["source_paths"][0] = get_absolute_path(source_path)
    args["target_path"] = get_absolute_path(target_path)
    args["output_path"] = get_absolute_path(output_path)

    if "face_swapper_model" in settings:
        args["face_swapper_model"] = settings["face_swapper_model"]

    if "face_enhancer_model" in settings:
        args["face_enhancer_model"] = settings["face_enhancer_model"]

    if "face_detector_model" in settings:
        args["face_detector_model"] = settings["face_detector_model"]

    with open(new_job_path, "w") as f:
        json.dump(job_data, f, indent=4)

    print(f"[+] Job file created at {new_job_path}")
    return job_id

def run_facefusion_with_job(job_id, execution_settings=None):
    if execution_settings is None:
        execution_settings = {}

    subprocess.run([
        "python", "facefusion.py",
        "job-submit",
        job_id
    ], cwd="facefusion", check=True)

    print(f"[+] Job {job_id} submitted.")

    command = ["python", "facefusion.py", "job-run", job_id]

    if "execution-device-id" in execution_settings:
        command += ["--execution-device-id", str(execution_settings["execution-device-id"])]
    if "execution-providers" in execution_settings:
        command += ["--execution-providers", execution_settings["execution-providers"]]
    if "execution-thread-count" in execution_settings:
        command += ["--execution-thread-count", str(execution_settings["execution-thread-count"])]
    if "execution-queue-count" in execution_settings:
        command += ["--execution-queue-count", str(execution_settings["execution-queue-count"])]

    subprocess.run(command, cwd="facefusion", check=True)

    print(f"[+] Job {job_id} executed with settings: {execution_settings}")

async def send_output_to_main_server(file_path):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            files = {'file': (os.path.basename(file_path), f, 'video/mp4')}
            response = await client.post(MAIN_SERVER_UPLOAD_URL, files=files)
    print(f"[+] Sent result to main server, status code: {response.status_code}")
    return response.status_code

@app.post("/run_ai/")
async def run_ai(file: UploadFile = File(...)):
    saved_filename = f"{file.filename}"
    saved_file_path = os.path.join(UPLOAD_FOLDER, saved_filename)

    with open(saved_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[+] Source image saved at {saved_file_path}")

    settings = {
        "face_swapper_model": "inswapper_128_fp16",
        "face_enhancer_model": "gfpgan_1.4",
        "face_detector_model": "scrfd"
    }
    execution_settings = {
        "execution-providers": "coreml"
    }
    output_file_path = os.path.join(OUTPUT_FOLDER, "output.mp4")

    job_id = create_job_from_basic(
        source_path=saved_file_path,
        target_path=TARGET_VIDEO_PATH,
        output_path=output_file_path,
        settings=settings
    )

    run_facefusion_with_job(job_id, execution_settings)

    await send_output_to_main_server(output_file_path)

    return {"status": "ok"}
