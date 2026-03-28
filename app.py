from flask import Flask, request, jsonify
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
    # Resize to 224x224 (what MobileNetV2 expects)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    
    return predicted_class, confidence

def get_spray_decision(predicted_class, confidence, temperature, humidity):
    # Base spray level from model prediction
    if predicted_class == 'Healthy':
        spray_level = 'none'
        spray_amount = 0
    elif predicted_class == 'Mild':
        spray_level = 'light'
        spray_amount = 30
    else:  # Severe
        spray_level = 'heavy'
        spray_amount = 100

    # Adjust based on temperature and humidity
    if predicted_class != 'Healthy':
        # High humidity makes disease spread faster → increase spray
        if humidity > 80:
            spray_amount = min(spray_amount + 20, 100)
        # Very high temperature stresses plant → increase spray slightly
        if temperature > 35:
            spray_amount = min(spray_amount + 10, 100)
        # Low humidity → reduce spray to avoid drying out plant
        if humidity < 30:
            spray_amount = max(spray_amount - 10, 0)

    return spray_level, spray_amount

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    temperature = float(request.form.get('temperature', 25))
    humidity = float(request.form.get('humidity', 50))
    
    # Read and predict
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)