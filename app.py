from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('fruits_classifier.h5')

# Define the classes
classes = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']  # replace with your actual class names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the file to the server
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Make a prediction
        preds = model.predict(img)
        class_idx = np.argmax(preds[0])
        class_label = classes[class_idx]

        # Return the result
        return jsonify({'prediction': class_label})

if __name__ == '__main__':
    # Ensure the 'uploads' folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # For production deployment
    app.run(host='0.0.0.0', port=8000)
