import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# ---------------- DIRECTORIES ---------------- #
data_dir = '../data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

model_dir = '../model'
reports_dir = '../reports'

os.makedirs(model_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# ---------------- MODEL ---------------- #
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------- DATA GENERATORS ---------------- #
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 🔥 CHECK CLASS MAPPING
print("\n🔥 CLASS INDICES (VERY IMPORTANT):")
print(train_generator.class_indices)

# ---------------- CALLBACKS ---------------- #
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    os.path.join(model_dir, 'brain_tumor_model.h5'),
    save_best_only=True,
    monitor='val_accuracy'
)

# ---------------- TRAIN ---------------- #
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # ✅ Reduced epochs
    callbacks=[early_stop, checkpoint]
)

# ---------------- SAVE TRAINING PLOT ---------------- #
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.savefig(os.path.join(reports_dir, 'training_history.png'))
plt.close()

# ---------------- EVALUATION ---------------- #
print("\n🔍 Evaluating on Validation Set...")

y_true = val_generator.classes
y_pred_prob = model.predict(val_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

# ---------------- CONFUSION MATRIX ---------------- #
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Tumor', 'Tumor'],
    yticklabels=['No Tumor', 'Tumor']
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'))
plt.close()

print("\n✅ Training complete.")
print("Model saved to: ../model/brain_tumor_model.h5")
print("Plots saved to: ../reports/")
