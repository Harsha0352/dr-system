import os

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5  # 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative
learning_rate = 0.0001
EPOCHS = 10

# Determine the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dr_model.h5")
DATASET_DIR = "d:/DRDS_HARSHA/dr_unified_v2/dr_unified_v2" # Keep for ref but mostly unused in deploy

def get_tf():
    """Lazy import tensorflow to prevent startup crashes"""
    import tensorflow as tf
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    return tf

def build_model():
    tf = get_tf()
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    # Skipped for deployment stability
    pass 

def load_trained_model():
    print(f"Looking for model at: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        try:
            # Check file size to detect LFS pointer
            file_size_bytes = os.path.getsize(MODEL_PATH)
            file_size_mb = file_size_bytes / (1024 * 1024)
            print(f"DEBUG: Found model file. Size in bytes: {file_size_bytes}")
            print(f"DEBUG: Found model file. Size in MB: {file_size_mb:.2f} MB")
            
            if file_size_mb < 0.1:
                print(f"DEBUG: LFS Pointer detected (Size {file_size_bytes} bytes). Downloading actual model...")
                try:
                    import requests
                    url = "https://raw.githubusercontent.com/Harsha0352/dr-system/main/backend/dr_model.h5"
                    print(f"Downloading from {url}...")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(MODEL_PATH, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print("Download complete! Verifying size...")
                    file_size_bytes = os.path.getsize(MODEL_PATH)
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    print(f"New file size: {file_size_mb:.2f} MB")
                    
                    if file_size_mb < 10:
                         print("❌ Downloaded file is still too small. LFS might be broken on GitHub side too.")
                         return None
                         
                except Exception as dl_err:
                    print(f"❌ Failed to download model: {dl_err}")
                    return None
            
            print("Loading saved model...")
            tf = get_tf()
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print("No saved model found. Switching to Mock Mode.")
        return None

if __name__ == "__main__":
    pass
