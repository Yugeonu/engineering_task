from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
from PIL import Image, ImageEnhance, ImageFilter
import unicodedata
import os
import sys
import numpy as np
import cv2
from collections import defaultdict
# import sys
# import json
# import zipfile

def preprocess_image(image):
    """이미지를 전처리하여 TrOCR에 적합하게 만듭니다."""
    # 흑백 변환
    image = image.convert('L')

    # 이미지 크기 조정 (예: 2배로 확대)
    image = image.resize((image.width * 2, image.height * 2))

    # 노이즈 제거
    image = image.filter(ImageFilter.MedianFilter(size=3))

    # 명암비 조정
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    
    # 다시 RGB로 변환
    image = image.convert('RGB')

    return image


# 현재 작업 디렉토리를 CRAFT 디렉토리가 있는 위치로 변경합니다.
file_path = 'CRAFT/test.py' #test의 상대경로
dir = os.path.dirname(os.path.abspath(file_path))
original_dir = os.getcwd()
os.chdir(dir)

# sys.path에 해당 디렉토리를 추가하여 모듈 임포트 문제를 해결합니다.
sys.path.insert(0, dir)

try:
    from test import detecting
    detected_coordinates = detecting()
    detected_coordinates, image_list = detected_coordinates[:-1], detected_coordinates[-1]
    image_len = len(image_list)
    
    for i in range(image_len): # 사진 경로가 바뀌면 수정해야함
        image_list[i] = 'CRAFT/'+image_list[i]
except:
    detected_coordinates, image_list = False, False
    
finally:
    # 작업을 마친 후 원래 작업 디렉토리로 복원합니다.
    os.chdir(original_dir)
    sys.path.pop(0)

if detected_coordinates:
    print("coordinates detected, Let's process")
    # print(detected_coordinates[0][0])
    # print(image_list)
else:
    print('failed with loading image and coordinates')
    exit(0)
    
processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")

full_sentence = [str(i+1)+' ' for i in range(image_len)]
full_tokken = []

for i in range(image_len):
    image = Image.open(image_list[i])
    box_list = detected_coordinates[i]
    result = []
    for j in range(len(box_list)):
        x_min, y_min = map(int, box_list[j][0])
        x_max, y_max = map(int, box_list[j][2])
        
        cropped_img = image.crop((x_min, y_min, x_max, y_max))
        cropped_img = preprocess_image(cropped_img)
        
        # # PIL 이미지를 numpy 배열로 변환
        # croppimg = np.array(cropped_img)
        
        # # OpenCV는 BGR 형식을 사용하므로 RGB 형식을 BGR로 변환
        # croppimg = cv2.cvtColor(croppimg, cv2.COLOR_RGB2BGR)
        
        # # 크롭된 이미지 표시
        # cv2.imshow(f'Cropped Image {i+1}-{j+1}', croppimg)
        # cv2.waitKey(0)  # 키보드 입력이 있을 때까지 창을 유지
        # cv2.destroyAllWindows()

            # TrOCR을 사용하여 텍스트 추출
        pixel_values = processor(cropped_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_length=64)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text = unicodedata.normalize("NFC", generated_text)
        
        result.append([generated_text, (x_min,y_min,x_max,y_max)])
        
        #print(i+1, '사진', j+1, '단어', generated_text)
               
    #print(result)
    
    # y축 범위 내에서 같은 줄로 간주할 허용 오차
    y_tolerance = 20

    # y축 기준으로 단어들을 그룹화하기 위한 딕셔너리 초기화
    lines = defaultdict(list)

    # OCR 결과를 y축 기준으로 그룹화
    for word, (x_min, y_min, x_max, y_max) in result:
        key = (y_min // y_tolerance) * y_tolerance
        lines[key].append((x_min, word))

    # 각 그룹 내에서 x축 기준으로 정렬하고 문장 생성
    sorted_lines = {}
    for key in lines:
        sorted_lines[key] = ' '.join([word for x_min, word in sorted(lines[key])])

    # 결과 출력
    for key in sorted(sorted_lines):
        full_tokken.append(sorted_lines[key])
        full_sentence[i] = full_sentence[i]+sorted_lines[key]
        
print(full_sentence)
print(full_tokken)