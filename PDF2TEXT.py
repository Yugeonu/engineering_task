import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2
import os
import re

# Tesseract 실행 파일의 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Tesseract 데이터 파일(tessdata)의 경로 설정
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'


def extract_text_from_pdf(pdf_path):
    """PyMuPDF를 사용하여 PDF에서 텍스트 블록 순서대로 추출"""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))  # y, x 순으로 정렬
        for block in blocks:
            text += block[4]
    return text


def correct_encoding_issues(text):
    """텍스트의 인코딩 문제 수정"""
    corrected_text = text.replace('�', ' ')
    # 추가적인 인코딩 수정이 필요할 경우 여기에 추가
    corrected_text = re.sub(r'\s+', ' ', corrected_text)
    return corrected_text


def pdf_to_jpgs(pdf_path, output_dir, zoom_x=3.0, zoom_y=3.0):
    """PDF 파일을 고해상도 JPG 이미지로 변환"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    jpg_files = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        matrix = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output_path = os.path.join(output_dir, f"page_{page_num + 1}.jpg")
        img.save(output_path, "JPEG")
        jpg_files.append(output_path)
        print(f"Saved {output_path}")  # 디버그 메시지 추가
    return jpg_files


def preprocess_image(image_path):
    """이미지 전처리"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # PIL을 사용하여 이미지를 열고 numpy array로 변환
    try:
        image = Image.open(image_path)
        image = np.array(image)
    except Exception as e:
        raise ValueError(f"Failed to read the image file: {image_path}. Error: {e}")

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # PIL 이미지를 RGB로 읽기 때문에 RGB2GRAY로 변경

    # 이미지 크기 조정
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(resized, None, 30, 7, 21)

    # 대비 및 밝기 조정
    alpha = 2.0  # 대비
    beta = 50  # 밝기
    adjusted = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)

    # 샤프닝 필터 적용
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)

    # 이진화
    _, binary = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def image_to_text(image_path):
    """이미지에서 Tesseract OCR을 사용하여 텍스트 추출"""
    preprocessed_image = preprocess_image(image_path)
    custom_config = r'--oem 3 --psm 3'
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config, lang='kor+eng')
    return text


def save_text(text, output_file):
    """추출된 텍스트를 파일에 저장"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)


def main(pdf_path, output_dir, output_file):
    """전체 과정 실행"""
    text = extract_text_from_pdf(pdf_path)
    corrected_text = correct_encoding_issues(text)

    if corrected_text.strip() and not '�' in corrected_text:
        # 텍스트 기반 PDF의 경우
        save_text(corrected_text, output_file)
    else:
        # 이미지 기반 PDF의 경우 또는 텍스트 인코딩 문제가 있는 경우
        jpg_files = pdf_to_jpgs(pdf_path, output_dir, zoom_x=3.0, zoom_y=3.0)
        texts = [image_to_text(jpg) for jpg in jpg_files]
        save_text("\n".join(texts), output_file)
    print(f"Text extracted and saved to {output_file}")


if __name__ == "__main__":
    pdf_path = r'C:\Users\samsung\Desktop\python 가상환경\OCR_python\test1.pdf'  # 입력 PDF 파일 경로
    output_dir = r'C:\Users\samsung\Desktop\python 가상환경\OCR_python\output_images'  # JPG 파일을 저장할 디렉토리 경로
    output_file = r'C:\Users\samsung\Desktop\python 가상환경\OCR_python\extracted_text.txt'  # 출력 텍스트 파일 경로
    main(pdf_path, output_dir, output_file)
