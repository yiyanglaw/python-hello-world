# server.py

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64

# Load the trained model

model = tf.keras.models.load_model('./api/advanced_action_recognition_model.h5')
label_mapping = np.load('./api/label_mapping.npy', allow_pickle=True).item()

label_mapping = {v: k for k, v in label_mapping.items()}

# Constants
IMG_SIZE = (224, 224)

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    frame = frame / 255.0
    return frame

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the base64 image from the request
        data = request.get_json()
        image_data = data['image']

        # Decode the image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make prediction
        input_frame = np.expand_dims(processed_frame, axis=0)
        predictions = model.predict(input_frame)
        predicted_index = np.argmax(predictions)

        # Map prediction to label
        if predicted_index in label_mapping:
            predicted_label = label_mapping[predicted_index]
        else:
            predicted_label = "Unknown"

        # Determine if the action is eating_medicine or not
        if predicted_label == 'eating_medicine':
            display_label = 'eating_medicine'
        else:
            display_label = 'not_eating_medicine'

        return jsonify({'prediction': display_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
