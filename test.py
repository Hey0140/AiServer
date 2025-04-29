# test_create_job.py
from AiServer import create_job_from_basic  # 네 함수가 있는 파일명으로 수정해
import os

if __name__ == "__main__":
    # 테스트용 settings 값
    settings = {
        "face_swapper_model": "inswapper_128_fp16",
        "face_detector_model": "scrfd"
    }

    # 테스트용 입력 값
    source_path = "test_source.png"
    target_path = "target.mp4"
    output_path = "test_output.mp4"

    # job.json 생성
    job_path = create_job_from_basic(
        source_path=source_path,
        target_path=target_path,
        output_path=output_path,
        settings=settings
    )

    print(f"Job created at: {job_path}")
    #
    # # 결과 파일 한번 열어보기
    # if os.path.exists(job_path):
    #     print("[+] Job JSON created successfully. Preview:")
    #     with open(job_path, "r") as f:
    #         print(f.read())
    # else:
    #     print("[-] Failed to create job.json.")
