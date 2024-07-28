from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_folder, image_format='jpeg'):
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i + 1}.{image_format}')
        image.save(image_path, image_format.upper())
        image_paths.append(image_path)
    return image_paths

# 사용 예시
pdf_path = 'Image_folder/testImage.pdf'
output_folder = 'CRAFT/image_folder/'
os.makedirs(output_folder, exist_ok=True)
image_paths = pdf_to_images(pdf_path, output_folder, image_format='jpeg')
print(f'Converted PDF to images: {image_paths}')
