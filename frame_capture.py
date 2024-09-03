# 슬라이드 넘어갈 떄 첫번째 화면만 캡쳐
# real use
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 동영상 파일 경로
video_path = r"C:\LLM_CONTEST2024\engineering_task\C1W1L04 Supervised Learning 2.mp4"
output_folder = r"C:\LLM_CONTEST2024\engineering_task\slide_cloud"

# 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 동영상 파일을 열기
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 5)  # 10초마다 프레임 추출

buffer = []
frame_count = 0
saved_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        if buffer:
            # 이전 프레임과 SSIM 비교
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_last = cv2.cvtColor(buffer[-1], cv2.COLOR_BGR2GRAY)
            score = ssim(gray_last, gray_current)

            if score < 0.8:
                # 첫 번째 프레임을 저장하고 버퍼 초기화
                saved_frame_path = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.png")
                cv2.imwrite(saved_frame_path, buffer[0])
                saved_frame_count += 1
                buffer.clear()

        # 현재 프레임을 버퍼에 추가
        buffer.append(frame)

    frame_count += 1

# 남은 버퍼의 첫 번째 프레임 저장
if buffer:
    saved_frame_path = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.png")
    cv2.imwrite(saved_frame_path, buffer[0])

cap.release()


