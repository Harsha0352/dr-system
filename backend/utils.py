from PIL import Image
import numpy as np
import io
from tensorflow.keras.applications.resnet50 import preprocess_input

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
    # ResNet50 preprocessing (Mean subtraction, BGR alignment handled by Keras)
    image_array = preprocess_input(image_array)
    return image_array
