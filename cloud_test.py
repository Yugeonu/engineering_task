import subprocess

# `jinyoung` 브랜치에서 frame_capture.py 실행
file_path = "C:\\LLM_CONTEST2024\\engineering_task\\frame_capture.py"
subprocess.run(['git', 'checkout', 'jinyoung'], check=True)
subprocess.run(['python', file_path], check=True)
print("Step 1 completed")

# 파일 강제로 저장 (다시 쓰기)
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print(f"{file_path} has been saved.")

# Git 상태 확인 및 변경 사항이 있으면 커밋 및 푸시
status_result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
if status_result.stdout.strip():
    # 변경 사항이 있으면 커밋하고 푸시
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run(['git', 'commit', '-m', 'Completed step 1: frame_capture.py execution'], check=True)
    subprocess.run(['git', 'push', 'origin', 'minha'], check=True)
    print("Changes committed and pushed for step 1")
else:
    print("No changes to commit")

# `minha` 브랜치에서 image_ssim.py 실행
try:
    subprocess.run(['git', 'checkout', 'minha'], check=True)
    subprocess.run(['python', 'image_ssim.py'], check=True)
    print("Step 2 completed")
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")

print("Finished processing")
