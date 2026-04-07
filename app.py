from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('plant_model_checkpoint.h5')
class_names = ['Healthy', 'Mild', 'Severe']

def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return predicted_class, confidence

def get_spray_decision(predicted_class, confidence, temperature, humidity):
    
    # Base spray from infection level
    if predicted_class == 'Healthy':
        spray_amount = 0
    elif predicted_class == 'Mild':
        spray_amount = 30
    else:  # Severe
        spray_amount = 100

    # Temperature adjustments
    if temperature > 40:
        spray_amount = min(spray_amount + 20, 100)
    elif temperature > 35:
        spray_amount = min(spray_amount + 10, 100)
    elif temperature < 15:
        spray_amount = max(spray_amount - 10, 0)

    # Humidity adjustments
    if humidity > 80:
        spray_amount = min(spray_amount + 20, 100)
    elif humidity > 60:
        spray_amount = min(spray_amount + 10, 100)
    elif humidity < 30:
        spray_amount = max(spray_amount - 10, 0)

    # Determine spray level label from final amount
    if spray_amount == 0:
        spray_level = 'none'
    elif spray_amount <= 30:
        spray_level = 'light'
    elif spray_amount <= 60:
        spray_level = 'moderate'
    else:
        spray_level = 'heavy'

    return spray_level, spray_amount
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    temperature = float(request.form.get('temperature', 25))
    humidity = float(request.form.get('humidity', 50))
    img = Image.open(io.BytesIO(file.read()))
    predicted_class, confidence = predict_image(img)
    spray_level, spray_amount = get_spray_decision(
        predicted_class, confidence, temperature, humidity
    )
    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence * 100, 2),
        'temperature': temperature,
        'humidity': humidity,
        'spray_level': spray_level,
        'spray_amount': spray_amount
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running!'})

@app.route('/test')
def test_page():
    return send_from_directory('.', 'test.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
