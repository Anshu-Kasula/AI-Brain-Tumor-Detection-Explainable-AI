import os
import shutil
import random

source_dir = r"archive\Dataset\Brain Tumor MRI images"
target_dir = "data"

classes = {
    "tumor": "tumor",
    "no_tumor": "no_tumor"
}

for cls, new_name in classes.items():
    src_path = os.path.join(source_dir, cls)
    train_path = os.path.join(target_dir, "train", new_name)
    val_path = os.path.join(target_dir, "val", new_name)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    images = os.listdir(src_path)
    random.shuffle(images)

    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(src_path, img), train_path)

    for img in val_images:
        shutil.copy(os.path.join(src_path, img), val_path)

print("Dataset split completed successfully!")
