import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Config ---
MODEL_PATH = 'trash_classifier_mobilenetv2.h5'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Make sure this matches the training class names
CLASS_NAMES = sorted([d for d in os.listdir('garbage') if os.path.isdir(os.path.join('garbage', d))])

# --- Load the trained model ---
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- Function to predict a single image ---
def predict_image(img_path):
    try:
        # Load image
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0        # rescale
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence

# --- Continuous prompt loop ---
print("\nType the image path to predict or 'exit' to quit.")
while True:
    img_path = input("Enter image path: ").strip()
    if img_path.lower() == 'exit':
        print("Exiting...")
        break

    pred_class, conf = predict_image(img_path)
    if pred_class is not None:
        print(f"Predicted class: {pred_class}, Confidence: {conf:.2f}")
