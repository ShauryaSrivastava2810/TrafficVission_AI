import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model_path = 'traffic_image_classification_model.h5'

class TrafficImagePrediction:
    def __init__(self, model_path):
        self.model = load_model(model_path)  # Use load_model directly from tensorflow.keras.models
        self.img_height = 150
        self.img_width = 150

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def predict_image(self, image_path):
        img_array = self.preprocess_image(image_path)
        prediction = self.model.predict(img_array)
        return 'Traffic' if prediction[0] >= 0.95 else 'No Traffic'
