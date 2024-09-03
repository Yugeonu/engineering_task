import subprocess

# `jinyoung` 브랜치에서 frame_capture.py 실행
subprocess.run(['git', 'checkout', 'jinyoung'], check=True)
subprocess.run(['python', r"C:\LLM_CONTEST2024\engineering_task\frame_capture.py"], check=True)
print("step1 compledted")

# `minha` 브랜치에서 image_ssim.py 실행
subprocess.run(['git', 'checkout', 'minha'], check=True)
subprocess.run(['python', r"C:\LLM_CONTEST2024\engineering_task\image_ssim.py"], check=True)

# print("Finished processing")
