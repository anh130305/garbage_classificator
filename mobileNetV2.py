import os
import random
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

# --- GPU Setup ---
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
BASE_DIR = 'dataset_split'      # Train/test folder
IMG_HEIGHT = 224                # MobileNetV2 default input size
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = len(os.listdir(os.path.join(BASE_DIR, 'train')))

# --- Data Generators with Augmentation ---
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

print(train_generator.class_indices)


test_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- Build MobileNetV2 Model ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model to keep pre-trained features
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Train Model ---
steps_per_epoch = math.ceil(train_generator.samples / BATCH_SIZE)
validation_steps = math.ceil(test_generator.samples / BATCH_SIZE)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS
)

# --- Optional: Fine-tune some top layers ---
# Unfreeze last few layers of base_model for fine-tuning
# base_model.trainable = True
# for layer in base_model.layers[:-20]:   # freeze first layers
#     layer.trainable = False
# model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
# history_finetune = model.fit(...)

# --- Save Model ---
model.save('trash_classifier_mobilenetv22.h5')
print("Model saved as trash_classifier_mobilenetv22.h5")
