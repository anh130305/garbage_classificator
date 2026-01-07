import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Config ---
MODEL_PATH = 'trash_classifier_model.h5'
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = sorted([d for d in os.listdir('garbage') if os.path.isdir(os.path.join('garbage', d))])  # same as training

# --- Load the trained model ---
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- Function to predict a single image ---
def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0        # rescale
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence

# --- Example usage ---
test_image_path = 'dataset_split/test/plastic/plastic2.jpg'  # Replace with your test image
pred_class, conf = predict_image(test_image_path)
print(f"Predicted class: {pred_class}, Confidence: {conf:.2f}")
