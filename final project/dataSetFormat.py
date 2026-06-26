import os
import cv2
import mediapipe as mp
from pathlib import Path

# Paths
DATASET_DIR = "datasets/lettersDF/DFtrainLetter"   # adjust if needed
OUTPUT_IMAGES = Path("LetterDataset/lettersDF/DFtrainLetter/images")
OUTPUT_LABELS = Path("LetterDataset/lettersDF/DFtrainLetter/labels")

# Define class mapping
classes = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "nothing","space"
]
class_to_id = {cls: i for i, cls in enumerate(classes)}

# Create output dirs
for split in ["train", "val"]:
    (OUTPUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_LABELS / split).mkdir(parents=True, exist_ok=True)

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def create_yolo_label(img_path, class_id, save_img_path, save_label_path):
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape

    # Detect hand
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        # Get bounding box from landmarks
        x_coords = [lm.x for lm in results.multi_hand_landmarks[0].landmark]
        y_coords = [lm.y for lm in results.multi_hand_landmarks[0].landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Convert to YOLO format
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # Save image
        cv2.imwrite(str(save_img_path), img)

        # Save label
        with open(save_label_path, "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

def process_dataset():
    # Simple split: 80% train, 20% val
    for class_name in classes:
        folder = Path(DATASET_DIR) / class_name
        if not folder.exists():
            continue

        files = list(folder.glob("*.jpg"))
        split_idx = int(0.8 * len(files))

        for i, img_path in enumerate(files):
            split = "train" if i < split_idx else "val"
            save_img_path = OUTPUT_IMAGES / split / img_path.name
            save_label_path = OUTPUT_LABELS / split / (img_path.stem + ".txt")

            create_yolo_label(
                img_path,
                class_to_id[class_name],
                save_img_path,
                save_label_path
            )

process_dataset()
print("✅ Labels created in YOLO format.")
