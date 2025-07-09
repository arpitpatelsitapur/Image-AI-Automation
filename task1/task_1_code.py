import cv2
import numpy as np
import os
import shutil

def get_thermal_box_from_aligned_image(aligned_path):
    """
    Detect the non-black bounding box from a manually aligned thermal image.
    Returns the (x, y, w, h) box and canvas dimensions (W, H).
    """
    aligned = cv2.imread(aligned_path)
    if aligned is None:
        raise ValueError("Could not load aligned thermal image")

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    canvas_h, canvas_w = aligned.shape[:2]
    return (x, y, w, h), canvas_w, canvas_h

def place_new_thermal_in_box(raw_thermal_path, box, canvas_w, canvas_h, output_path):
    """
    Resize a new raw thermal image to fit exactly in the reference box
    and paste it into a black canvas of same dimensions as aligned thermal.
    """
    thermal = cv2.imread(raw_thermal_path)
    if thermal is None:
        print(f"Skipped (could not load): {raw_thermal_path}")
        return

    x, y, w, h = box

    resized_thermal = cv2.resize(thermal, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[y:y+h, x:x+w] = resized_thermal

    cv2.imwrite(output_path, canvas)
    print(f"[✓] Aligned thermal saved: {output_path}")

def copy_rgb_image(base_id, input_dir, output_dir):
    """
    Copies the corresponding RGB image (XXXX_Z.JPG) to output directory.
    """
    rgb_name = f"{base_id}_Z.JPG"
    src = os.path.join(input_dir, rgb_name)
    dst = os.path.join(output_dir, rgb_name)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"[✓] RGB image copied: {dst}")
    else:
        print(f"[!] RGB not found for: {base_id}")

if __name__ == "__main__":
    input_dir = "input_images"
    output_dir = "task_1_output"

    os.makedirs(output_dir, exist_ok=True)

    reference_aligned_path = os.path.join(output_dir, "DJI_20250530123037_0003_AT.JPG")
    thermal_box, canvas_w, canvas_h = get_thermal_box_from_aligned_image(reference_aligned_path)

    for filename in os.listdir(input_dir):
        if filename.endswith("_T.JPG"):
            base_id = filename.replace("_T.JPG", "")
            input_thermal_path = os.path.join(input_dir, filename)
            output_thermal_path = os.path.join(output_dir, f"{base_id}_AT.JPG")

            # Align thermal and copy RGB
            place_new_thermal_in_box(input_thermal_path, thermal_box, canvas_w, canvas_h, output_thermal_path)
            copy_rgb_image(base_id, input_dir, output_dir)