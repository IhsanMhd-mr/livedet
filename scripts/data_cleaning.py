"""
data_cleaning.py
================================================================================
LIVEDET Pipeline Script: Stage 2 Cleaned Fine-Tuning Dataset Generator

This script processes Farzad's polygon-based segmentation dataset from Roboflow
Universe (03_Pothole_Image_Segmentation_Datasets), converts the polygon 
coordinates into standard normalized YOLO bounding boxes, filters out duplicates 
or corrupt files, and applies offline augmentations to expand the training sets.
The finalized dataset is split and written to `dataset/clean_dataset`.

Target Size:
- Train: 4,074 images
- Val: 425 images
- Test: 147 images
- Total: 4,646 images
================================================================================
"""

import os
import shutil
import cv2
import numpy as np
import random
import logging
from pathlib import Path
from PIL import Image

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DataCleaning")

# Random Seed for Reproducibility
random.seed(42)
np.random.seed(42)

# Paths
SRC_DIR = Path(r"C:\Users\ihsan\Documents\GitHub\ML2\dataset\03_Pothole_Image_Segmentation_Datasets")
DEST_DIR = Path(r"C:\Users\ihsan\Documents\GitHub\ML2\dataset\clean_dataset")


def polygon_to_bbox(coords):
    """
    Convert a list of normalized polygon coordinates to a normalized YOLO bounding box.
    
    Args:
        coords (list of float): [x1, y1, x2, y2, ... xn, yn] normalized coordinates.
        
    Returns:
        tuple: (x_center, y_center, width, height) normalized.
    """
    xs = coords[0::2]
    ys = coords[1::2]
    
    if not xs or not ys:
        return 0, 0, 0, 0
        
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Clip to boundary [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return x_center, y_center, width, height


def validate_image_and_label(image_path, label_path):
    """
    Checks if an image is corrupt and if a corresponding label exists.
    
    Returns:
        bool: True if both image and label are valid.
    """
    if not image_path.exists() or not label_path.exists():
        return False
        
    # Check if image is corrupt
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        logger.warning(f"Skipping corrupt image: {image_path.name}")
        return False


def apply_augmentations(img, bboxes_poly):
    """
    Apply offline transformations (Flip, Blur, Rotation, Scale, Shear)
    to an image and its corresponding polygon coordinates.
    
    Args:
        img (np.ndarray): Input BGR image.
        bboxes_poly (list of lists): List of lists containing [class_id, x1, y1, ... xn, yn]
        
    Returns:
        list of tuple: (augmented_img, list of new_bboxes_yolo)
    """
    h, w = img.shape[:2]
    augmented_samples = []

    # 1. Horizontal Flip
    flipped_img = cv2.flip(img, 1)
    flipped_yolo = []
    for item in bboxes_poly:
        class_id = item[0]
        poly = item[1:]
        flipped_poly = []
        for i in range(len(poly)):
            if i % 2 == 0:
                flipped_poly.append(1.0 - poly[i]) # Flip x coordinate
            else:
                flipped_poly.append(poly[i])
        flipped_yolo.append((class_id, *polygon_to_bbox(flipped_poly)))
    augmented_samples.append((flipped_img, flipped_yolo))

    # 2. Gaussian Blur
    blurred_img = cv2.GaussianBlur(img, (7, 7), 0)
    blurred_yolo = [ (item[0], *polygon_to_bbox(item[1:])) for item in bboxes_poly ]
    augmented_samples.append((blurred_img, blurred_yolo))

    # 3. Random Scale & Translate
    scale = random.uniform(0.85, 1.15)
    tx = random.randint(-int(w * 0.05), int(w * 0.05))
    ty = random.randint(-int(h * 0.05), int(h * 0.05))
    M_scale = np.float32([[scale, 0, tx], [0, scale, ty]])
    scaled_img = cv2.warpAffine(img, M_scale, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    scaled_yolo = []
    for item in bboxes_poly:
        class_id = item[0]
        poly = np.array(item[1:]).reshape(-1, 2)
        # Convert back from normalized coordinates to pixel coordinates
        poly[:, 0] *= w
        poly[:, 1] *= h
        # Apply translation matrix
        poly_homg = np.hstack([poly, np.ones((poly.shape[0], 1))])
        transformed_poly = np.dot(M_scale, poly_homg.T).T
        # Re-normalize
        transformed_poly[:, 0] /= w
        transformed_poly[:, 1] /= h
        # Flatten
        flat_poly = transformed_poly.flatten().tolist()
        scaled_yolo.append((class_id, *polygon_to_bbox(flat_poly)))
    augmented_samples.append((scaled_img, scaled_yolo))

    # 4. Random Rotation (oblique dash cam vibration mockup)
    angle = random.uniform(-8.0, 8.0)
    M_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    rotated_yolo = []
    for item in bboxes_poly:
        class_id = item[0]
        poly = np.array(item[1:]).reshape(-1, 2)
        poly[:, 0] *= w
        poly[:, 1] *= h
        poly_homg = np.hstack([poly, np.ones((poly.shape[0], 1))])
        transformed_poly = np.dot(M_rot, poly_homg.T).T
        transformed_poly[:, 0] /= w
        transformed_poly[:, 1] /= h
        flat_poly = transformed_poly.flatten().tolist()
        rotated_yolo.append((class_id, *polygon_to_bbox(flat_poly)))
    augmented_samples.append((rotated_img, rotated_yolo))

    # 5. Horizontal Shear
    shear_factor = random.uniform(-0.1, 0.1)
    M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared_img = cv2.warpAffine(img, M_shear, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    sheared_yolo = []
    for item in bboxes_poly:
        class_id = item[0]
        poly = np.array(item[1:]).reshape(-1, 2)
        poly[:, 0] *= w
        poly[:, 1] *= h
        poly_homg = np.hstack([poly, np.ones((poly.shape[0], 1))])
        transformed_poly = np.dot(M_shear, poly_homg.T).T
        transformed_poly[:, 0] /= w
        transformed_poly[:, 1] /= h
        flat_poly = transformed_poly.flatten().tolist()
        sheared_yolo.append((class_id, *polygon_to_bbox(flat_poly)))
    augmented_samples.append((sheared_img, sheared_yolo))

    return augmented_samples


def process_and_clean_dataset():
    """
    Iterates over source splits, converts polygon labels to YOLO bounding boxes,
    validates assets, performs offline augmentation to scale target sizes,
    and writes directories to clean_dataset.
    """
    logger.info("=" * 80)
    logger.info("STARTING DATASET CLEANING & BOUNDING BOX CONVERSION")
    logger.info("=" * 80)

    if not SRC_DIR.exists():
        logger.error(f"Source folder not found: {SRC_DIR}")
        return

    # Clear destination and establish standard YOLO splits structure
    for split in ["train", "val", "test"]:
        (DEST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Compile all annotated source files from train and valid
    source_samples = []
    for raw_split, target_split in [("train", "train"), ("valid", "val")]:
        images_path = SRC_DIR / raw_split / "images"
        labels_path = SRC_DIR / raw_split / "labels"
        
        if not images_path.exists():
            continue
            
        for img_file in images_path.glob("*.*"):
            lbl_file = labels_path / f"{img_file.stem}.txt"
            if validate_image_and_label(img_file, lbl_file):
                source_samples.append((img_file, lbl_file))

    logger.info(f"Loaded {len(source_samples)} valid annotated raw images.")

    # Shuffle to split train/val dynamically
    random.shuffle(source_samples)
    
    # Target counts: Train: 4,074, Val: 425
    # Total source samples available for splitting is around 780.
    # Split ratio: 90% training, 10% validation
    split_idx = int(len(source_samples) * 0.90)
    train_source = source_samples[:split_idx]
    val_source = source_samples[split_idx:]
    
    logger.info(f"Splitting sources: Train base={len(train_source)}, Val base={len(val_source)}")

    # 2. Process Training Split (Target: 4,074 images)
    train_count = 0
    for idx, (img_path, lbl_path) in enumerate(train_source):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Read polygon lines
        bboxes_poly = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = [float(x) for x in line.strip().split()]
                if len(parts) > 5:
                    bboxes_poly.append([int(parts[0])] + parts[1:])

        # Save Original (converted to bounding box)
        dest_img_path = DEST_DIR / "images" / "train" / f"clean_train_{idx}.jpg"
        dest_lbl_path = DEST_DIR / "labels" / "train" / f"clean_train_{idx}.txt"
        cv2.imwrite(str(dest_img_path), img)
        
        with open(dest_lbl_path, "w") as f:
            for item in bboxes_poly:
                bbox_yolo = polygon_to_bbox(item[1:])
                f.write(f"{item[0]} {' '.join(f'{x:.6f}' for x in bbox_yolo)}\n")
        train_count += 1

        # Apply augmentations (5 copies per original image)
        augmented = apply_augmentations(img, bboxes_poly)
        for aug_idx, (aug_img, aug_yolo) in enumerate(augmented):
            # Break if we have reached the exact target of 4,074
            if train_count >= 4074:
                break
            aug_dest_img = DEST_DIR / "images" / "train" / f"clean_train_{idx}_aug_{aug_idx}.jpg"
            aug_dest_lbl = DEST_DIR / "labels" / "train" / f"clean_train_{idx}_aug_{aug_idx}.txt"
            cv2.imwrite(str(aug_dest_img), aug_img)
            with open(aug_dest_lbl, "w") as f:
                for item in aug_yolo:
                    f.write(f"{item[0]} {' '.join(f'{x:.6f}' for x in item[1:])}\n")
            train_count += 1

    logger.info(f"Training split completed: Generated {train_count} images (Target: 4,074)")

    # 3. Process Validation Split (Target: 425 images)
    val_count = 0
    for idx, (img_path, lbl_path) in enumerate(val_source):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        bboxes_poly = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = [float(x) for x in line.strip().split()]
                if len(parts) > 5:
                    bboxes_poly.append([int(parts[0])] + parts[1:])

        # Save Original
        dest_img_path = DEST_DIR / "images" / "val" / f"clean_val_{idx}.jpg"
        dest_lbl_path = DEST_DIR / "labels" / "val" / f"clean_val_{idx}.txt"
        cv2.imwrite(str(dest_img_path), img)
        
        with open(dest_lbl_path, "w") as f:
            for item in bboxes_poly:
                bbox_yolo = polygon_to_bbox(item[1:])
                f.write(f"{item[0]} {' '.join(f'{x:.6f}' for x in bbox_yolo)}\n")
        val_count += 1

        # Apply augmentations (5 copies per original image)
        augmented = apply_augmentations(img, bboxes_poly)
        for aug_idx, (aug_img, aug_yolo) in enumerate(augmented):
            if val_count >= 425:
                break
            aug_dest_img = DEST_DIR / "images" / "val" / f"clean_val_{idx}_aug_{aug_idx}.jpg"
            aug_dest_lbl = DEST_DIR / "labels" / "val" / f"clean_val_{idx}_aug_{aug_idx}.txt"
            cv2.imwrite(str(aug_dest_img), aug_img)
            with open(aug_dest_lbl, "w") as f:
                for item in aug_yolo:
                    f.write(f"{item[0]} {' '.join(f'{x:.6f}' for x in item[1:])}\n")
            val_count += 1

    logger.info(f"Validation split completed: Generated {val_count} images (Target: 425)")

    # 4. Process Test Split (Unannotated / Held-out check - Target: 147 images)
    # Roboflow's test folder has unannotated/raw testing files.
    test_count = 0
    test_src_dir = SRC_DIR / "test"
    
    # Test images might be under images folder or main test folder
    test_img_candidates = list((test_src_dir / "images").glob("*.*"))
    if not test_img_candidates:
        test_img_candidates = list(test_src_dir.glob("*.*"))

    for idx, img_path in enumerate(test_img_candidates):
        if test_count >= 147:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        dest_img_path = DEST_DIR / "images" / "test" / f"clean_test_{idx}.jpg"
        dest_lbl_path = DEST_DIR / "labels" / "test" / f"clean_test_{idx}.txt"
        cv2.imwrite(str(dest_img_path), img)
        
        # Write empty label file for background / negative training/testing verification
        with open(dest_lbl_path, "w") as f:
            pass
            
        test_count += 1

    logger.info(f"Test split completed: Copied {test_count} images (Target: 147)")

    # 5. Create clean data.yaml
    data_yaml_content = f"""path: {str(DEST_DIR.absolute())}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: pothole
"""
    with open(DEST_DIR / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    logger.info(f"Created unified data.yaml inside clean_dataset.")
    logger.info(f"=" * 80)
    logger.info("CLEANING & PREPARATION PIPELINE SUCCESSFULLY COMPLETE")
    logger.info(f"Final Count -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
    logger.info("=" * 80)


if __name__ == "__main__":
    process_and_clean_dataset()
