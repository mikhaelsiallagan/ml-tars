import os
import io
import psycopg2

from dotenv import load_dotenv
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
from google.cloud import storage
from datetime import datetime

from uuid import uuid4

load_dotenv()

app = Flask(__name__)

# Configure Google Cloud Storage
bucket_name = os.getenv("BUCKET_NAME") 
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
storage_client = storage.Client.from_service_account_json(credentials_path)
bucket = storage_client.bucket(bucket_name)


LATEST_PREDICTION_UUID = "00000000-0000-0000-0000-000000000001"

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Connect to GCP SQL 
def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=5432,
        host='34.41.150.181'
    )
    return conn

def preprocess_image(image):
    width, height = 800, 800
    return image.resize((width, height))

# Load the YOLO model
model_path = "best-ir9.onnx"
model = YOLO(model_path, task='detect')

class_labels = {0: 'metal', 1: 'paper', 2: 'plastic', 3: 'background'}

@app.route('/add-bin', methods=['POST'])
def add_bin():
    data = request.get_json()
    bin_type = data.get('bin_type')
    if not bin_type:
        return jsonify({'error': 'bin_type is required'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO bins (bin_type, fullness_level) VALUES (%s, %s) RETURNING bin_id",
        (bin_type, 0)
    )
    bin_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'bin_id': bin_id, 'bin_type': bin_type}), 201

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = preprocess_image(img)

    # Save image locally for YOLO prediction
    upload_dir = './static/uploads'
    os.makedirs(upload_dir, exist_ok=True)
    img_path = os.path.join(upload_dir, 'temp_image.jpg')
    img.save(img_path)

    results = model.predict(img_path, save=False, imgsz=800, conf=0.25)

    predictions = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            if confidence > 0.1:
                class_label = class_labels.get(class_id, 'unknown')
                predictions.append({
                    'class_id': class_id,
                    'class_label': class_label,
                    'confidence': confidence
                })

    # Generate a timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detect-results/detection_{timestamp}.jpg"

    # Upload processed image to Google Cloud Storage with timestamped filename
    blob = bucket.blob(filename)
    blob.upload_from_filename(img_path)
    blob.make_public() 
    image_url = blob.public_url

    # Map class labels to bin IDs
    label_to_bin_id = {
        'metal': os.getenv("METAL_ID"),
        'plastic': os.getenv("PLASTIC_ID"),
        'paper': os.getenv("PAPER_ID")
    }

    # Get bin_id based on detected class label
    detected_label = predictions[0]['class_label'] if predictions else 'unknown'
    bin_id = label_to_bin_id.get(detected_label, None)

    if bin_id is None:
        return jsonify({'error': f'No bin found for label: {detected_label}'}), 400

    # Insert into database
    conn = get_db_connection()
    cursor = conn.cursor()
    scan_id = str(uuid4())
    cursor.execute(
        "INSERT INTO scans (scan_id, timestamp, detected_type, image_url, bin_id) VALUES (%s, %s, %s, %s, %s)",
        (scan_id, datetime.utcnow(), detected_label, image_url, bin_id)
    )
    for pred in predictions:
        cursor.execute(
            "INSERT INTO predictions (prediction_id, scan_id, class_id, class_label, confidence) VALUES (%s, %s, %s, %s, %s)",
            (str(uuid4()), scan_id, pred['class_id'], pred['class_label'], pred['confidence'])
        )

    cursor.execute(
        """
        INSERT INTO latest_prediction (latest_id, prediction_id, scan_id, timestamp, detected_type, image_url)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (latest_id) DO UPDATE
        SET prediction_id = EXCLUDED.prediction_id,
            timestamp = EXCLUDED.timestamp,
            detected_type = EXCLUDED.detected_type,
            image_url = EXCLUDED.image_url,
            scan_id = EXCLUDED.scan_id
        """,
        (LATEST_PREDICTION_UUID, str(uuid4()), scan_id, datetime.utcnow(), detected_label, image_url)
    )


    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'predictions': predictions, 'image_url': image_url, 'bin_id': bin_id})



@app.route('/get-data', methods=['GET'])
def get_data():
    bin_id = request.args.get('bin_id')
    bins_data = []

    conn = get_db_connection()
    cursor = conn.cursor()

    if bin_id:
        # Retrieve specific bin details from the bins table
        cursor.execute("SELECT * FROM bins WHERE bin_id = %s", (bin_id,))
        bin = cursor.fetchone()
        bins_data = {'bin': bin}
    else:
        # Retrieve all bins data from the bins table
        cursor.execute("SELECT * FROM bins")
        bins_data = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(bins_data)

@app.route('/add-data', methods=['POST'])
def add_fullness_data():
    data = request.get_json()

    bin_id = data.get('bin_id')
    fullness_level_cm = data.get('fullness_level_cm')

    # Validate input data
    if not bin_id:
        return jsonify({'error': 'bin_id is required'}), 400
    if fullness_level_cm is None:
        return jsonify({'error': 'fullness_level_cm is required'}), 400
    if not isinstance(fullness_level_cm, int) or fullness_level_cm < 0:
        return jsonify({'error': 'fullness_level_cm must be a non-negative integer'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Insert fullness data into fullness_logs
        cursor.execute(
            """
            INSERT INTO fullness_logs (log_id, bin_id, timestamp, fullness_level_cm) 
            VALUES (%s, %s, %s, %s)
            """,
            (str(uuid4()), bin_id, datetime.utcnow(), fullness_level_cm)
        )

        # Update the bins table with the latest fullness level and timestamp
        cursor.execute(
            """
            UPDATE bins 
            SET fullness_level = %s, last_updated = %s
            WHERE bin_id = %s
            """,
            (fullness_level_cm, datetime.utcnow(), bin_id)
        )

        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        return jsonify({'error': 'Database error', 'details': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({'message': 'Fullness data added successfully', 'bin_id': bin_id, 'fullness_level_cm': fullness_level_cm}), 201

@app.route('/add-status', methods=['POST'])
def add_status():
    data = request.get_json()
    status = data.get('status')  # Should be True/False

    # Validate input
    if status is None or not isinstance(status, bool):
        return jsonify({'error': 'Invalid or missing "status". Must be True or False.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Use a fixed UUID for single-row logic
        fixed_uuid = '123e4567-e89b-12d3-a456-426614174000'  # Replace this with a valid UUID

        # Insert or update the status in the status_updates table
        cursor.execute(
            """
            INSERT INTO status_updates (status_id, timestamp, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (status_id) DO UPDATE
            SET timestamp = EXCLUDED.timestamp, status = EXCLUDED.status
            """,
            (fixed_uuid, datetime.utcnow(), status)  # Use the fixed UUID
        )
        conn.commit()

    except psycopg2.Error as e:
        conn.rollback()
        return jsonify({'error': 'Database error', 'details': str(e)}), 500

    finally:
        cursor.close()
        conn.close()

    return jsonify({'status': status}), 201

@app.route('/get-status', methods=['GET'])
def get_status():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Use the fixed UUID for the single row
        fixed_uuid = '123e4567-e89b-12d3-a456-426614174000'  # Ensure this matches the UUID in the /add-status endpoint

        # Fetch the single row based on the fixed UUID
        cursor.execute(
            """
            SELECT status_id, timestamp, status
            FROM status_updates
            WHERE status_id = %s
            """,
            (fixed_uuid,)
        )
        latest_status = cursor.fetchone()

        if latest_status:
            # Update the status to FALSE
            cursor.execute(
                """
                UPDATE status_updates
                SET status = FALSE, timestamp = NOW()
                WHERE status_id = %s
                """,
                (fixed_uuid,)
            )
            conn.commit()

            # Return only the original status
            result = {'status': latest_status[2]}  # Original status before the update
        else:
            result = {'error': 'No status entry found'}

    except psycopg2.Error as e:
        conn.rollback()
        return jsonify({'error': 'Database error', 'details': str(e)}), 500

    finally:
        cursor.close()
        conn.close()

    return jsonify(result), 200

@app.route('/get-predict', methods=['GET'])
def get_predict():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Query the fixed UUID
        fixed_uuid = LATEST_PREDICTION_UUID
        cursor.execute(
            """
            SELECT prediction_id, scan_id, timestamp, detected_type, image_url
            FROM latest_prediction
            WHERE latest_id = %s
            """,
            (fixed_uuid,)
        )
        latest_prediction = cursor.fetchone()

        if latest_prediction:
            # Format the result
            result = {
                'prediction_id': latest_prediction[0],
                'scan_id': latest_prediction[1],
                'timestamp': latest_prediction[2],
                'detected_type': latest_prediction[3],
                'image_url': latest_prediction[4]
            }
        else:
            result = {'error': 'No prediction data available'}

    except psycopg2.Error as e:
        return jsonify({'error': 'Database error', 'details': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4500)
