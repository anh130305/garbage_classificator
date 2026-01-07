import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Config ---
MODEL_PATH = 'trash_classifier_mobilenetv2.h5'
IMG_HEIGHT = 224
IMG_WIDTH = 224
TEST_DIR = 'dataset_split/test'

# --- Category grouping ---
CATEGORY_GROUPS = {
    'glass': ['brown-glass', 'green-glass', 'white-glass'],
    'paper_cardboard': ['paper', 'cardboard'],
    # other classes remain the same
    'battery': ['battery'],
    'biological': ['biological'],
    'clothes': ['clothes'],
    'metal': ['metal'],
    'plastic': ['plastic'],
    'shoes': ['shoes'],
    'trash': ['trash']
}

# Flattened list of all classes in the dataset
ALL_CLASSES = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])

# --- Load the trained model ---
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- Function to map original class to grouped class ---
def map_to_group(cls):
    for group_name, members in CATEGORY_GROUPS.items():
        if cls in members:
            return group_name
    return cls  # fallback, should not happen

# --- Function to predict a single image ---
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = ALL_CLASSES[np.argmax(predictions)]
    return map_to_group(predicted_class)

# --- Evaluation ---
total_images = 0
correct_predictions = 0

# Track per-group accuracy
group_counts = {group: 0 for group in CATEGORY_GROUPS}
group_correct = {group: 0 for group in CATEGORY_GROUPS}

for cls in ALL_CLASSES:
    class_dir = os.path.join(TEST_DIR, cls)
    if not os.path.exists(class_dir):
        continue
    group_label = map_to_group(cls)
    
    for fname in os.listdir(class_dir):
        img_path = os.path.join(class_dir, fname)
        if not os.path.isfile(img_path):
            continue

        pred_group = predict_image(img_path)
        total_images += 1
        group_counts[group_label] += 1

        if pred_group == group_label:
            correct_predictions += 1
            group_correct[group_label] += 1

# --- Print results ---
overall_acc = correct_predictions / total_images if total_images > 0 else 0
print(f"\nOverall accuracy (grouped): {overall_acc:.4f} ({correct_predictions}/{total_images})\n")

print("Accuracy per group:")
for group_name in CATEGORY_GROUPS:
    if group_counts[group_name] == 0:
        acc = 0
    else:
        acc = group_correct[group_name] / group_counts[group_name]
    print(f"  {group_name}: {acc:.4f} ({group_correct[group_name]}/{group_counts[group_name]})")
