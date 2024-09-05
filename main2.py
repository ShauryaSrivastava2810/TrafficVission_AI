from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
import os
import collections
import time
import cv2
from TrafficImagePrediction import TrafficImagePrediction as tip
from TrafficVideoPrediction import TrafficVideoPrediction as tvp
from PIL import Image
import imageio

app = Flask(__name__)
CORS(app)

# Paths to models and Haar Cascade
IMAGE_MODEL_PATH = 'traffic_image_classification_model.h5'
VIDEO_MODEL_PATH = 'video_classification_model.keras'
HAAR_CASCADE_PATH = 'haarcascade_car.xml'

# Initialize models
image_model = tip(IMAGE_MODEL_PATH)
video_model = tvp(VIDEO_MODEL_PATH)


def process_image(image_path):
    result = image_model.predict_image(image_path)
    return result


def process_video(video_path):
    count = 0
    ncount = 0
    results = video_model.predict_video(video_path)
    for result in results:
        if result.lower() == "traffic":
            count += 1
        elif result.lower() == "no traffic":
            ncount += 1
    return count, ncount


def generate_graph_data(video_path, time_interval):
    cap = cv2.VideoCapture(video_path)
    car_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    car_count_dict = collections.defaultdict(int)
    start_time = time.time()

    while True:
        ret, frames = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        current_time = time.time()
        elapsed_time = current_time - start_time
        interval = int(elapsed_time // time_interval)

        car_count_dict[interval] += len(cars)

    cap.release()

    return car_count_dict


@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    media_type = request.form['media']

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    if media_type == 'image':
        result = process_image(file_path)
        return jsonify({'result': result})
    elif media_type == 'video':
        count, ncount = process_video(file_path)
        result = 'Traffic' if count > ncount else 'No Traffic' if count < ncount else 'Neutral'
        return jsonify({'result': result})


@app.route('/graph', methods=['POST'])
def graph_data():
    file = request.files['file']
    time_interval = int(request.form['interval'])

    video_path = os.path.join('uploads', file.filename)
    car_count_dict = generate_graph_data(video_path, time_interval)

    car_count_dict_formatted = {f"{interval}": count for interval, count in sorted(car_count_dict.items())}
    return jsonify(car_count_dict_formatted)


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
