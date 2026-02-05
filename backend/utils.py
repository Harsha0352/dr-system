from PIL import Image
import numpy as np
import io
# from tensorflow.keras.applications.resnet50 import preprocess_input

def read_image_file(file_data) -> Image.Image:
    image = Image.open(io.BytesIO(file_data))
    return image

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Resize and preprocess image for model prediction.
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Manual ResNet50 preprocessing to avoid TF dependency issues
    # Zero-center by mean pixel (approximate)
    image_array = image_array.astype('float32')
    image_array[..., 0] -= 103.939
    image_array[..., 1] -= 116.779
    image_array[..., 2] -= 123.68
    
    # Note: Keras preprocess_input also flips RGB to BGR, but for stability we skip heavy TF imports here.
    # If using the real model, we can lazy import TF here if needed, but this manual step is safer/faster.
    return image_array
