import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import math

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU found. Using GPU for training.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

# --- Config ---
DATASET_DIR = 'garbage'         # Original dataset with 12 class folders
BASE_DIR = 'dataset_split'      # Folder to create train/test split
IMG_HEIGHT = 128 #224
IMG_WIDTH = 128 #224
BATCH_SIZE = 32 
EPOCHS = 20
TEST_SPLIT = 0.1             # 10% of images for testing
SKIP_SPLIT = True

# --- Create train/test folders ---
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

if not SKIP_SPLIT:
    for split in ['train', 'test']:
        for class_name in classes:
            os.makedirs(os.path.join(BASE_DIR, split, class_name), exist_ok=True)

    # --- Split data ---
    for class_name in classes:
        class_folder = os.path.join(DATASET_DIR, class_name)
        images = os.listdir(class_folder)
        random.shuffle(images)
        split_index = int(len(images) * (1 - TEST_SPLIT))
        train_images = images[:split_index]
        test_images = images[split_index:]

        for img in train_images:
            shutil.copy(os.path.join(class_folder, img), os.path.join(BASE_DIR, 'train', class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_folder, img), os.path.join(BASE_DIR, 'test', class_name, img))

    print("Data split into train and test successfully!")

# --- Data Generators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- Build Model ---
NUM_CLASSES = len(classes)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Train ---
steps_per_epoch = math.ceil(train_generator.samples / BATCH_SIZE)
validation_steps = math.ceil(test_generator.samples / BATCH_SIZE)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS
)

# --- Save Model ---
model.save('trash_classifier_model.h5')
print("Model saved as trash_classifier_model.h5")
