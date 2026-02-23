import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gradcam_utils import get_gradcam_heatmap, calculate_time_to_trust

# ---------------- LOAD MODEL ---------------- #
model = load_model('../model/brain_tumor_model.h5')

data_dir = '../data'
reports_dir = '../reports'
os.makedirs(reports_dir, exist_ok=True)

# ---------------- DATA GENERATOR ---------------- #
val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ---------------- PREDICTIONS ---------------- #
y_true = val_generator.classes
y_pred_prob = model.predict(val_generator).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_true, y_pred))

# ---------------- CONFUSION MATRIX ---------------- #
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Tumor', 'Tumor'],
            yticklabels=['No Tumor', 'Tumor'])
plt.title('Confusion Matrix')
plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'))
plt.close()

# ---------------- ROC CURVE ---------------- #
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(reports_dir, 'roc_curve.png'))
plt.close()

# ---------------- GRAD-CAM SAMPLE ---------------- #
sample_img_path = os.path.join(
    data_dir, 'val', 'tumor',
    os.listdir(os.path.join(data_dir, 'val', 'tumor'))[0]
)

img = tf.keras.preprocessing.image.load_img(sample_img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

# Generate heatmap
heatmap = get_gradcam_heatmap(model, img_array)

# Resize heatmap to match image size
heatmap = cv2.resize(heatmap, (224, 224))

# Normalize properly
heatmap = np.maximum(heatmap, 0)
if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)

heatmap_colored = cv2.applyColorMap(
    np.uint8(255 * heatmap),
    cv2.COLORMAP_JET
)

original_img = np.array(img)
original_img = np.uint8(original_img)

overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

plt.imshow(overlay)
plt.title('Sample Grad-CAM Heatmap')
plt.axis('off')
plt.savefig(os.path.join(reports_dir, 'sample_gradcam.png'))
plt.close()

# ---------------- TRUST SCORE ---------------- #
confidence = float(y_pred_prob[0])
time_to_trust, trust_score = calculate_time_to_trust(confidence)

print(f"Sample Time-to-Trust: {time_to_trust:.2f}")
print(f"Trust Score: {trust_score:.2f}")

# ---------------- PDF REPORT ---------------- #
pdf_path = os.path.join(reports_dir, 'evaluation_report.pdf')
c = canvas.Canvas(pdf_path, pagesize=letter)

c.drawString(100, 750, "Brain Tumor Detection Model Evaluation Report")
c.drawString(100, 730, f"Model Accuracy: {np.mean(y_pred == y_true):.2f}")
c.drawString(100, 710, f"ROC AUC: {roc_auc:.2f}")
c.drawString(100, 690, f"Sample Time-to-Trust: {time_to_trust:.2f}")
c.drawString(100, 670, f"Trust Score: {trust_score:.2f}")
c.drawString(100, 650, "Refer to saved PNG files for Confusion Matrix, ROC Curve, and Grad-CAM.")

c.save()

print("Evaluation complete. Check 'reports/' folder.")
