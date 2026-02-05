import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5  # 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative
learning_rate = 0.0001
# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU Detected: {len(gpus)} device(s) found. Training will be fast!")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected. Training might be slow.")

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5  # 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative
learning_rate = 0.0001
EPOCHS = 10

DATASET_DIR = "d:/DRDS_HARSHA/dr_unified_v2/dr_unified_v2"
MODEL_PATH = "d:/DRDS_HARSHA/dr-system/backend/dr_model.h5"

def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers (optional: fine-tuning needed for better results)
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        return None

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Train or Val directory missing.")
        return None

    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary' if NUM_CLASSES == 2 else 'sparse' # Assuming sparse categorical for 5 classes
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary' if NUM_CLASSES == 2 else 'sparse'
    )

    model = build_model()
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=50,  # LITE MODE: Limit to 50 batches (1600 images) per epoch for CPU speed
        validation_steps=10,
        epochs=EPOCHS
    )
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("No saved model found. Creating a new (untrained) one for demo purposes.")
        return build_model()

if __name__ == "__main__":
    train_model()
