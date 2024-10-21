import os
import io
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)


def preprocess_image(image):  # Fungsi untuk memproses gambar
    # Resize gambar jika diperlukan (ukuran default YOLO dapat diatur)
    width = 800
    height = 800
    image = image.resize((width, height))  # Optional resizing if needed
    return image


# Load the YOLO model
model_path = "best-ir9.onnx"  # Path model YOLOv8M
# model_path = "./models/YOLOv8S/best.onnx"  # Path model YOLOv8S
model = YOLO(model_path, task='detect')


# Define a mapping of class IDs to class labels
class_labels = {
    0: 'metal',
    1: 'paper',
    2: 'plastic',
    3: 'background',
}


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Baca gambar dari file upload
    img = Image.open(io.BytesIO(file.read())).convert('RGB')

    # Preprocess the image (resize if necessary)
    img = preprocess_image(img)

    # Create the uploads directory if it doesn't exist
    upload_dir = './static/uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the processed image temporarily for YOLO prediction
    img_path = os.path.join(upload_dir, 'temp_image.jpg')
    img.save(img_path)

    # Lakukan prediksi dengan YOLO
    results = model.predict(img_path, save=False, imgsz=800, conf=0.25)

    # Extract predictions: classes, confidences, and class labels
    predictions = []
    for result in results:
        for box in result.boxes:
            # Extract the class and confidence for each detection
            class_id = int(box.cls)  # Class index
            confidence = float(box.conf)  # Confidence score

            # Filter based on confidence threshold
            if confidence > 0.1:
                # Map the class_id to the class label
                class_label = class_labels.get(class_id, 'unknown')

                # Append class_id, class_label, and confidence to predictions
                predictions.append({
                    'class_id': class_id,
                    'class_label': class_label,
                    'confidence': confidence
                })

    # Return the predictions as JSON
    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4500)