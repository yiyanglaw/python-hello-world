from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
import base64

# Load the model and label mapping....
model = tf.keras.models.load_model('./api/advanced_action_recognition_model.h5')
label_mapping = np.load('./api/label_mapping.npy', allow_pickle=True).item()
label_mapping = {v: k for k, v in label_mapping.items()}

IMG_SIZE = (224, 224)

def preprocess_image(image_data):
    # Decode base64 image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

class handler(BaseHTTPRequestHandler):

    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            body = json.loads(post_data)
            image_data = body['image']
            input_frame = preprocess_image(image_data)

            # Make prediction
            predictions = model.predict(input_frame)
            predicted_index = np.argmax(predictions)

            # Map prediction to label
            if predicted_index in label_mapping:
                predicted_label = label_mapping[predicted_index]
            else:
                predicted_label = "Unknown"

            response = {
                'prediction': predicted_label
            }
            self._set_headers(200)
        except Exception as e:
            response = {
                'error': str(e)
            }
            self._set_headers(400)

        self.wfile.write(json.dumps(response).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=handler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    httpd.serve_forever()

if __name__ == "__main__":
    print(1)
    run()
