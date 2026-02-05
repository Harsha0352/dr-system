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
# Determine the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dr_model.h5")

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
    # ... (Train logic skipped for deployment context, keeping function signature for compatibility) ...
    pass 

def load_trained_model():
    print(f"Looking for model at: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        try:
            # Check file size to detect LFS pointer (usually < 1KB)
            file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"Found model file. Size: {file_size_mb:.2f} MB")
            
            if file_size_mb < 0.1:
                print("⚠️ File is too small (< 100KB). It might be a Git LFS pointer. Falling back to dummy model.")
                return build_model()
                
            print("Loading saved model...")
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Falling back to new (untrained) model for demo purposes.")
            return build_model()
    else:
        print("⚠️ No saved model found. Creating a new (untrained) one for demo purposes.")
        return build_model()

if __name__ == "__main__":
    train_model()
