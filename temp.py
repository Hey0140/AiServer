import os

from AiServerCuda import create_job_from_basic, run_facefusion_with_job, send_output_to_main_server

if __name__ == "__main__":
    UPLOAD_FOLDER = "uploads/"
    source = os.path.join(UPLOAD_FOLDER, "test.png")
    target = "target1.mp4"  # 대상 비디오
    output = "outputs/output_test.mp4"

    settings = {
        "face_swapper_model": "inswapper_128",  # 안정적
        "face_enhancer_model": "gfpgan_1.4",  # 실패 줄이기 위해 off
        "face_detector_model": "scrfd"
    }
    execution_settings = {
        "execution-providers": "cuda"
    }

    job_id, _ = create_job_from_basic(source, target, output)
    print("job_id:", job_id)
    run_facefusion_with_job(job_id, execution_settings)