import tensorflow as tf
import cv2
import numpy as np


class TrafficVideoPrediction:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.img_height = 128
        self.img_width = 128
        self.frames_per_batch = 10
        self.frames_accumulated = []

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (self.img_width, self.img_height))
        frame = frame / 255.0  # Normalize the frame
        return frame

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        predictions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.preprocess_frame(frame)
            self.frames_accumulated.append(processed_frame)

            if len(self.frames_accumulated) == self.frames_per_batch:
                batch_frames = np.array(self.frames_accumulated)
                batch_frames = np.expand_dims(batch_frames, axis=0)
                prediction = self.model.predict(batch_frames)
                predictions.extend(prediction.flatten())
                self.frames_accumulated = []

        # Process any remaining frames
        if len(self.frames_accumulated) > 0:
            batch_frames = np.array(self.frames_accumulated)
            batch_frames = np.expand_dims(batch_frames, axis=0)
            prediction = self.model.predict(batch_frames)
            predictions.extend(prediction.flatten())

        cap.release()

        # Convert predictions to 'Traffic' or 'No Traffic' based on threshold (0.5 in this case)
        prediction_labels = ['Traffic' if pred > 0.5 else 'No Traffic' for pred in predictions]
        return prediction_labels
