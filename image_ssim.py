import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

# 이미지 폴더 경로 및 결과 파일 경로
image_folder = r"C:\LLM_CONTEST2024\engineering_task\slide_cloud"
output_file = r"C:\LLM_CONTEST2024\engineering_task\slide_cloud_output"

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # 파일 이름 정렬

# SSIM 값을 저장할 파일 열기
with open(output_file, 'w') as file:
    for i in range(len(image_files) - 1):
        # 이미지 파일 경로
        img1_path = os.path.join(image_folder, image_files[i])
        img2_path = os.path.join(image_folder, image_files[i + 1])

        # 이미지 읽기
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # 이미지를 그레이스케일로 변환
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # SSIM 계산
        score, _ = ssim(gray1, gray2, full=True)

        # 결과를 텍스트 파일에 기록
        file.write(f"{image_files[i]} vs {image_files[i + 1]}: SSIM = {score:.4f}\n")

print(f"SSIM values saved to {output_file}")
