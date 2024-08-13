from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import cv2
import requests

# Initialize Flask app
app = Flask(__name__)

# Firebase Storage URLs
MODEL_URL = 'https://firebasestorage.googleapis.com/v0/b/firstfront.appspot.com/o/advanced_action_recognition_model.h5?alt=media'
LABEL_MAPPING_URL = 'https://firebasestorage.googleapis.com/v0/b/firstfront.appspot.com/o/label_mapping.npy?alt=media'
MODEL_FILENAME = 'advanced_action_recognition_model.h5'
LABEL_MAPPING_FILENAME = 'label_mapping.npy'
IMG_SIZE = (224, 224)

# Download the model and label mapping from Firebase Storage
def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded {destination}')
    else:
        raise Exception(f"Failed to download file from {url}")

# Download and load the model
download_file(MODEL_URL, MODEL_FILENAME)
download_file(LABEL_MAPPING_URL, LABEL_MAPPING_FILENAME)

# Load the model and label mapping
model = tf.keras.models.load_model(MODEL_FILENAME)
label_mapping = np.load(LABEL_MAPPING_FILENAME, allow_pickle=True).item()
label_mapping = {v: k for k, v in label_mapping.items()}

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    frame = frame / 255.0
    return frame

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)
    input_frame = np.expand_dims(processed_frame, axis=0)

    # Make prediction
    predictions = model.predict(input_frame)
    predicted_index = np.argmax(predictions)

    # Map prediction to label
    predicted_label = label_mapping.get(predicted_index, "Unknown")

    # Determine if the action is eating_medicine or not
    display_label = 'eating_medicine' if predicted_label == 'eating_medicine' else 'not_eating_medicine'

    return jsonify({'action': display_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
