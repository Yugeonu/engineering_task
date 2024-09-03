import cv2
import os
import numpy as np
import json
from skimage.metrics import structural_similarity as ssim

def get_timestamp_from_seconds(seconds):
    """Convert seconds to a timestamp string formatted as HH-MM-SS_MMM."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}-{minutes:02}-{seconds_int:02}_{milliseconds:03}"

def compare_frames(frame1, frame2):
    """Calculate the SSIM between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def extract_difference(first_frame, last_frame, threshold=50):
    """Extract the binary difference between two frames."""
    diff = cv2.absdiff(first_frame, last_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, binary_diff = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    return binary_diff

def is_white_image(image):
    """Check if the image is completely white."""
    white_pixels = np.sum(np.all(image == 255, axis=-1))
    total_pixels = image.shape[0] * image.shape[1]
    return white_pixels == total_pixels

def save_json(slide_info, output_dir):
    """Save the slide information to a JSON file."""
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    json_path = os.path.join(output_dir, f'{video_name}.json')
    with open(json_path, 'w') as json_file:
        json.dump(slide_info, json_file, indent=4, ensure_ascii=False)

def capture_frames(video_path, output_dir, capture_interval_sec=2, ssim_threshold=0.9):
    """Capture frames from a video at regular intervals and save them if SSIM is below a threshold."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval_count = int(capture_interval_sec * fps)
    os.makedirs(output_dir, exist_ok=True)

    ret, previous_frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return

    buffer = []
    frame_count = 0
    frame_number = 1  # Slide number starts from 1
    slide_info = []  # To hold JSON data for all slides
    writing_count = 0

    # Capture and save the first frame
    timestamp = get_timestamp_from_seconds(0)
    frame_dir = os.path.join(output_dir, f"{os.path.basename(video_path).rsplit('.', 1)[0]}_frame_{frame_number:05d}_{timestamp}")
    os.makedirs(frame_dir, exist_ok=True)
    slide_filename = os.path.join(frame_dir, f"slide_{frame_number:05d}_{timestamp}.png")
    cv2.imwrite(slide_filename, previous_frame)
    print(f"Captured frame {frame_number} at {timestamp} (Frame 0)")
    buffer.append(previous_frame)

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval_count == 0:
            similarity = compare_frames(previous_frame, current_frame)
            print(f"Frame {frame_count}, Similarity: {similarity:.4f}")

            if similarity < ssim_threshold:
                # Save the writings for the previous slide before moving to the next
                if buffer:
                    first_frame = buffer[0]
                    last_frame = buffer[-1]
                    diff = extract_difference(first_frame, last_frame)

                    white_background = np.ones_like(first_frame, dtype=np.uint8) * 255
                    white_background[diff != 0] = last_frame[diff != 0]

                    if not is_white_image(white_background):
                        writing_filename = os.path.join(frame_dir, f"writing_{frame_number:05d}.png")
                        cv2.imwrite(writing_filename, white_background)
                        writing_count += 1
                        print(f"Saved writing frame for slide at {frame_number}")

                # Save slide information to the JSON
                video_name = os.path.basename(video_path).rsplit('.', 1)[0]
                slide_info.append({
                    "video_name": video_name,
                    "frame_number": frame_number,
                    "slide_timestamp": timestamp,
                    "writing_count": writing_count
                })

                # Reset the writing count for the next slide
                writing_count = 0

                # Update to the new slide
                timestamp = get_timestamp_from_seconds(frame_count / fps)
                frame_number += 1  # Increment slide number
                frame_dir = os.path.join(output_dir, f"{os.path.basename(video_path).rsplit('.', 1)[0]}_frame_{frame_number:05d}_{timestamp}")
                os.makedirs(frame_dir, exist_ok=True)
                slide_filename = os.path.join(frame_dir, f"slide_{frame_number:05d}_{timestamp}.png")
                cv2.imwrite(slide_filename, current_frame)
                print(f"Captured frame {frame_number} at {timestamp} (Frame {frame_count}) with SSIM={similarity:.4f}")

                # Clear the buffer after saving
                buffer = []

            buffer.append(current_frame)
            previous_frame = current_frame

        frame_count += 1

    if buffer:
        # Handle remaining buffer if there are any frames left at the end
        first_frame = buffer[0]
        last_frame = buffer[-1]
        diff = extract_difference(first_frame, last_frame)
        white_background = np.ones_like(first_frame, dtype=np.uint8) * 255
        white_background[diff != 0] = last_frame[diff != 0]

        if not is_white_image(white_background):
            writing_filename = os.path.join(frame_dir, f"writing_{frame_number:05d}.png")
            cv2.imwrite(writing_filename, white_background)
            writing_count += 1
            print(f"Saved writing frame for slide at {frame_number}")

        # Save the last slide information to the JSON
        video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        slide_info.append({
            "video_name": video_name,
            "frame_number": frame_number,
            "slide_timestamp": timestamp,
            "writing_count": writing_count
        })

    cap.release()

    # Save the entire slide information to a JSON file
    save_json(slide_info, output_dir)

# # Usage example
video_path = r"/home/work/INC_LAB/JY/OCR/[PyTorch] Lab-03 Deeper Look at GD - 복사본.mp4"
output_dir = r"/home/work/INC_LAB/JY/OCR/OCR_TEST_con"
#
# capture_frames(video_path, output_dir, capture_interval_sec=2, ssim_threshold=0.9)
