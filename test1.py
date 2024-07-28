import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
from PIL import Image, ImageEnhance, ImageFilter
import unicodedata

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

# EasyOCR 초기화
reader = easyocr.Reader(['ko'])

# 이미지 경로
image_path = "/Users/yugeon-u/Downloads/test1.jpg"

# 이미지 열기
image = Image.open(image_path)

# 텍스트 영역 감지
result = reader.detect(image_path)

# 감지된 텍스트 영역 좌표 추출
boxes = result[0][0]  # 첫 번째 요소의 첫 번째 리스트가 텍스트 블록 좌표

# TrOCR 초기화
processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")

# 감지된 텍스트 영역 처리
for coords in boxes:
    x_max = int(max(coords[0], coords[2]))
    y_max = int(max(coords[1], coords[3]))
    x_min = int(min(coords[0], coords[2]))
    y_min = int(min(coords[1], coords[3]))
    
    # 텍스트 블록 자르기
    cropped_img = image.crop((x_min, y_min, x_max, y_max))
    
    # 이미지 전처리
    processed_img = preprocess_image(cropped_img)
    
    # TrOCR을 사용하여 텍스트 추출
    pixel_values = processor(processed_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=64)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = unicodedata.normalize("NFC", generated_text)
    
    # 결과 출력
    print(generated_text)
