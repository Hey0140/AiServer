# test_create_and_run_job.py

import subprocess
from AiServer import create_job_from_basic, run_facefusion_with_job
import os

if __name__ == "__main__":
    # 테스트용 settings 값
    settings = {
        "face_swapper_model": "inswapper_128_fp16",
        "face_enhancer_model": "gfpgan_1.4",
        "face_detector_model": "scrfd"
    }

    # 테스트용 execution settings
    execution_settings = {
        "execution-providers": "coreml"
    }

    # 테스트용 입력 경로
    source_path = "test_source.png"   # 얼굴이 있는 이미지
    target_path = "target.mp4"         # 타겟 비디오
    output_path = "test_output.mp4"    # 결과 저장 경로

    # (1) job 생성
    job_id = create_job_from_basic(
        source_path=source_path,
        target_path=target_path,
        output_path=output_path,
        settings=settings
    )

    print(f"[+] Job created with ID: {job_id}")

    # (2) FaceFusion 실행
    try:
        run_facefusion_with_job(job_id, execution_settings)
        print("[+] FaceFusion executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[-] FaceFusion execution failed: {e}")
