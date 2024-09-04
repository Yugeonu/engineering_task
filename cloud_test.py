import subprocess

# `jinyoung` 브랜치에서 frame_capture.py 실행
subprocess.run(['git', 'checkout', 'jinyoung'], check=True)
subprocess.run(['python', 'C:\\LLM_CONTEST2024\\engineering_task\\frame_capture.py'], check=True)
print("Step 1 completed")

# Git에 변경 사항 커밋 및 푸시
subprocess.run(['git', 'add', '.'], check=True)
subprocess.run(['git', 'commit', '-m', 'Completed step 1: frame_capture.py execution'], check=True)
subprocess.run(['git', 'push'], check=True)
print("Changes committed and pushed for step 1")

# `minha` 브랜치에서 image_ssim.py 실행
try:
    subprocess.run(['git', 'checkout', 'minha'], check=True)
    subprocess.run(['python', 'image_ssim.py'], check=True)
    print("Step 2 completed")
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")

print("Finished processing")
