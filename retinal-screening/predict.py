import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import sys

# Load model
model = tf.keras.models.load_model('models/retinal_model.h5')

# Class labels
class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return class_names[predicted_class], confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    prediction, confidence = predict_image(img_path)
    print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
