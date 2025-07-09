# task_2_code.py
# python 3.10.12

import os
import cv2
import numpy as np

def load_image_pairs(input_folder):
    """
    Scans the input folder and returns pairs of before and after images filenames.
    """
    before_images = [f for f in os.listdir(input_folder) if f.endswith('.jpg') and '~2' not in f and '~3' not in f]
    pairs = []
    for before_img in before_images:
        base_name = before_img[:-4]  # remove '.jpg'
        after_img = f"{base_name}~2.jpg"
        after_path = os.path.join(input_folder, after_img)
        before_path = os.path.join(input_folder, before_img)
        if os.path.exists(after_path):
            pairs.append((before_path, after_path))
    return pairs

def detect_missing_objects(before_img, after_img):
    """
    Detect and highlight missing objects in the after image compared to the before image.
    Returns list of bounding boxes (x, y, w, h).
    """

    # Convert both images to grayscale
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)

    # Compute the difference (before minus after)
    diff = cv2.absdiff(before_gray, after_gray)

    # Threshold the difference image to get regions that are significantly different
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Use morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    missing_bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter small areas to ignore noise
        if area > 250:
            x,y,w,h = cv2.boundingRect(cnt)
            missing_bboxes.append((x,y,w,h))

    return missing_bboxes

def annotate_after_image(after_img, bboxes):
    """
    Draw bounding boxes highlighting missing objects on the after image.
    """
    annotated_img = after_img.copy()
    for (x,y,w,h) in bboxes:
        cv2.rectangle(annotated_img, (x,y), (x+w, y+h), (0, 0, 255), 3) # red box
    return annotated_img

def main():
    input_folder = 'task_2_output'

    pairs = load_image_pairs(input_folder)
    if not pairs:
        print(f"No image pairs found in {input_folder}.")
        return

    for before_path, after_path in pairs:
        print(f"Processing pair:\n  Before: {before_path}\n  After: {after_path}")

        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)

        if before_img is None or after_img is None:
            print(f"Failed to load images: {before_path}, {after_path}")
            continue

        missing_bboxes = detect_missing_objects(before_img, after_img)

        annotated_img = annotate_after_image(after_img, missing_bboxes)

        # Save annotated after image using suffix ~3.jpg
        base_name = os.path.basename(before_path)[:-4]
        output_fname = f"{base_name}~3.jpg"
        output_path = os.path.join(input_folder, output_fname)

        cv2.imwrite(output_path, annotated_img)
        print(f"Saved annotated image: {output_path}")

if __name__ == "__main__":
    main()