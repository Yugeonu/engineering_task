import cv2
import numpy as np

# EAST 모델 파일 경로
east_model_path = 'frozen_east_text_detection.pb'

# 이미지 파일 경로
image_path = "/Users/yugeon-u/Downloads/test1.jpg"

# 이미지 읽기
image = cv2.imread(image_path)
original_image = image.copy()

# EAST 모델 로드
net = cv2.dnn.readNet(east_model_path)

# 이미지 크기 조정
(height, width) = image.shape[:2]
new_height = ((height // 32) + 1) * 32
new_width = ((width // 32) + 1) * 32
blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)

# 텍스트 디텍션 실행
(scores, geometry) = net.forward(["scores", "geometry"])

# 후처리: 텍스트 영역 감지 및 결과 출력
def decode_predictions(scores, geometry):
    # 코드에서 후처리 및 디텍션 박스 추출 부분을 작성합니다.
    pass

boxes, confidences = decode_predictions(scores, geometry)

# 감지된 텍스트 영역을 이미지에 표시
for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(original_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow('Text Detection', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
