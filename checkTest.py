# test.py
import sys
import os
from AiServerCuda import run_facefusion_with_job

if __name__ == "__main__":
    job_id = sys.argv[1]
    execution_settings = {
        "execution-providers": "cuda"
    }
    run_facefusion_with_job(job_id, execution_settings)
