from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from scipy.ndimage import label

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

def preprocess_image(image_data):
    # Remove the data URL prefix
    image_data = image_data.split(',')[1]
    
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale and resize to 28x28
    image = image.convert('L')
    image = image.resize((28, 28), Image.LANCZOS)
    
    # Invert image (MNIST is white digit on black background)
    image = Image.eval(image, lambda x: 255 - x)
    
    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array.reshape(1, 28, 28, 1)
    image_array = image_array.astype('float32') / 255.0
    
    return image_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_digit])

        # Confidence threshold
        if confidence < 0.5:
            return jsonify({
                'error': 'Could not confidently recognize a digit. Please draw a single, clear digit (0-9).'
            }), 400
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 