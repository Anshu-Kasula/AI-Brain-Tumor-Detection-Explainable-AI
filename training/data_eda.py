import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from collections import Counter

# Set up directories
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
reports_dir = 'reports'
os.makedirs(reports_dir, exist_ok=True)

# Function to get class distribution
def get_class_distribution(directory):
    classes = ['tumor', 'no_tumor']
    counts = {}
    for cls in classes:
        path = os.path.join(directory, cls)
        if os.path.exists(path):
            counts[cls] = len(os.listdir(path))
        else:
            counts[cls] = 0
    return counts

# EDA: Class distribution
train_counts = get_class_distribution(train_dir)
val_counts = get_class_distribution(val_dir)

print("Train Distribution:", train_counts)
print("Val Distribution:", val_counts)

# Visualize class distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()), ax=ax[0])
ax[0].set_title('Train Class Distribution')
sns.barplot(x=list(val_counts.keys()), y=list(val_counts.values()), ax=ax[1])
ax[1].set_title('Validation Class Distribution')
plt.savefig(os.path.join(reports_dir, 'class_distribution.png'))
plt.show()

# Sample images visualization
def plot_sample_images(directory, num_samples=5):
    classes = ['tumor', 'no_tumor']
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 6))
    for i, cls in enumerate(classes):
        path = os.path.join(directory, cls)
        if os.path.exists(path):
            images = os.listdir(path)[:num_samples]
            for j, img_name in enumerate(images):
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path).resize((224, 224))
                axes[i, j].imshow(img)
                axes[i, j].set_title(f'{cls} Sample {j+1}')
                axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'sample_images.png'))
    plt.show()

plot_sample_images(train_dir)

# Basic stats: Image sizes (optional, for variability check)
image_sizes = []
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            image_sizes.append(img.size)

sizes_df = pd.DataFrame(image_sizes, columns=['Width', 'Height'])
print(sizes_df.describe())
sizes_df.to_csv(os.path.join(reports_dir, 'image_sizes.csv'))

print("EDA complete. Check 'reports/' for saved plots and CSV.")