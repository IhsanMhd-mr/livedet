import os
import shutil
import yaml
from pathlib import Path

print("=" * 80)
print("DATASET REORGANIZATION & MERGING")
print("=" * 80)

# Paths
MY_DATASET = Path(r"C:\Users\ihsan\Documents\GitHub\ML2\dataset\My Dataset")
ANNOTATED_DATASET = Path(r"C:\Users\ihsan\Documents\GitHub\ML2\dataset\Dataset_Annotated")
AUGMENTED_DATASET = Path(r"C:\Users\ihsan\Documents\GitHub\ML2\dataset\Dataset_Aug\DetectPot_AUG_Split")

# Output paths
OUTPUT_BINARY = Path(r"C:\Users\ihsan\Documents\GitHub\ML2\dataset\01_BINARY_POTHOLE_DETECTION")
OUTPUT_DETAILED = Path(r"C:\Users\ihsan\Documents\GitHub\ML2\dataset\02_DETAILED_CRACKS_ANNOTATION")

def copy_recursive(src, dst):
    """Copy directory and return file count"""
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src)
            dest_file = dst / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_file)
            count += 1
    return count

print("\n" + "=" * 80)
print("STEP 1: REORGANIZE BINARY DATASET (My Dataset)")
print("=" * 80)

# Create binary detection dataset structure
(OUTPUT_BINARY / "images" / "train").mkdir(parents=True, exist_ok=True)
(OUTPUT_BINARY / "images" / "val").mkdir(parents=True, exist_ok=True)
(OUTPUT_BINARY / "labels" / "train").mkdir(parents=True, exist_ok=True)
(OUTPUT_BINARY / "labels" / "val").mkdir(parents=True, exist_ok=True)

# Copy images (binary dataset doesn't have labels, just folder structure)
train_count = 0
for category in ["pothole", "plain"]:
    src = MY_DATASET / "train" / category
    if src.exists():
        for img_file in src.glob("*.*"):
            shutil.copy2(img_file, OUTPUT_BINARY / "images" / "train" / img_file.name)
            train_count += 1

test_count = 0
for category in ["pothole", "plain"]:
    src = MY_DATASET / "test" / category
    if src.exists():
        for img_file in src.glob("*.*"):
            shutil.copy2(img_file, OUTPUT_BINARY / "images" / "val" / img_file.name)
            test_count += 1

print(f"✅ Training images: {train_count}")
print(f"✅ Validation images: {test_count}")

# Create YAML for binary dataset
binary_yaml = {
    'path': str(OUTPUT_BINARY.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': {0: 'pothole'}
}

with open(OUTPUT_BINARY / "data.yaml", 'w') as f:
    yaml.dump(binary_yaml, f, default_flow_style=False)

print(f"✅ YAML created: {OUTPUT_BINARY / 'data.yaml'}")
print(f"✅ Binary Dataset Ready: {train_count + test_count} total images")

print("\n" + "=" * 80)
print("STEP 2: MERGE ANNOTATED + AUGMENTED DATASETS (Detailed/Cracks)")
print("=" * 80)

# Create detailed detection dataset structure
(OUTPUT_DETAILED / "images" / "train").mkdir(parents=True, exist_ok=True)
(OUTPUT_DETAILED / "images" / "val").mkdir(parents=True, exist_ok=True)
(OUTPUT_DETAILED / "labels" / "train").mkdir(parents=True, exist_ok=True)
(OUTPUT_DETAILED / "labels" / "val").mkdir(parents=True, exist_ok=True)

# Copy Annotated dataset (images/labels)
anno_train_img = copy_recursive(
    ANNOTATED_DATASET / "images" / "train",
    OUTPUT_DETAILED / "images" / "train"
)
anno_train_lbl = copy_recursive(
    ANNOTATED_DATASET / "labels" / "train",
    OUTPUT_DETAILED / "labels" / "train"
)

anno_val_img = copy_recursive(
    ANNOTATED_DATASET / "images" / "test",
    OUTPUT_DETAILED / "images" / "val"
)
anno_val_lbl = copy_recursive(
    ANNOTATED_DATASET / "labels" / "test",
    OUTPUT_DETAILED / "labels" / "val"
)

print(f"✅ Annotated Train Images: {anno_train_img}")
print(f"✅ Annotated Train Labels: {anno_train_lbl}")
print(f"✅ Annotated Val Images: {anno_val_img}")
print(f"✅ Annotated Val Labels: {anno_val_lbl}")

# Copy Augmented dataset (merge with annotated)
aug_train_img = copy_recursive(
    AUGMENTED_DATASET / "train" / "images",
    OUTPUT_DETAILED / "images" / "train"
)
aug_train_lbl = copy_recursive(
    AUGMENTED_DATASET / "train" / "labels",
    OUTPUT_DETAILED / "labels" / "train"
)

aug_val_img = copy_recursive(
    AUGMENTED_DATASET / "valid" / "images",
    OUTPUT_DETAILED / "images" / "val"
)
aug_val_lbl = copy_recursive(
    AUGMENTED_DATASET / "valid" / "labels",
    OUTPUT_DETAILED / "labels" / "val"
)

print(f"✅ Augmented Train Images: {aug_train_img}")
print(f"✅ Augmented Train Labels: {aug_train_lbl}")
print(f"✅ Augmented Val Images: {aug_val_img}")
print(f"✅ Augmented Val Labels: {aug_val_lbl}")

# Create YAML for detailed dataset
detailed_yaml = {
    'path': str(OUTPUT_DETAILED.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 4,
    'names': {
        0: 'Pot',
        1: 'AllCrack',
        2: 'LongCrack',
        3: 'LatCrack'
    }
}

with open(OUTPUT_DETAILED / "data.yaml", 'w') as f:
    yaml.dump(detailed_yaml, f, default_flow_style=False)

print(f"✅ YAML created: {OUTPUT_DETAILED / 'data.yaml'}")

total_train = anno_train_img + aug_train_img
total_val = anno_val_img + aug_val_img
total_images = total_train + total_val

print(f"✅ Detailed Dataset Ready: {total_images} total images")
print(f"   - Training: {total_train}")
print(f"   - Validation: {total_val}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n✅ 1️⃣ BINARY DATASET: {OUTPUT_BINARY}")
print(f"   Purpose: Quick pothole identification (yes/no)")
print(f"   Images: {train_count + test_count}")

print(f"\n✅ 2️⃣ DETAILED DATASET: {OUTPUT_DETAILED}")
print(f"   Purpose: Fine-tuned crack detection & bounding boxes")
print(f"   Images: {total_images}")
print(f"   Classes: 4 (Pot, AllCrack, LongCrack, LatCrack)")

print("\n" + "=" * 80)
