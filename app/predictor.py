import cv2
import numpy as np
import tensorflow as tf
import os
from .pose_extractor import PoseExtractor
from .preprocessing import LandmarkPreprocessor, normalize_landmarks, extract_basic_features

class AlphabetPredictor:
    def __init__(self, model_path='models/alphabet/model.h5', scaler_path='models/alphabet/scaler.pkl'):
        # Get absolute paths to models relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, model_path)
        scaler_path = os.path.join(base_dir, scaler_path)
        landmarker_path = os.path.join(base_dir, 'models/hand_landmarker.task')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Alphabet model not found at {model_path}")

        # Load model and scaler
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.scaler = LandmarkPreprocessor.load(scaler_path)
        
        # Initialize PoseExtractor in IMAGE mode
        self.extractor = PoseExtractor(static_image_mode=True, model_path=landmarker_path)
        
        # Alphabet labels (Digits 1-9 then A-Z = 35 classes)
        self.labels = [str(i) for i in range(1, 10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    def predict(self, image_bytes):
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, "Invalid image data"

        # Extract landmarks
        landmarks, _ = self.extractor.extract_from_frame(image)
        
        if landmarks is None:
            return None, "No hand detected"

        # Preprocess
        # 1. Normalize landmarks (relative to wrist, scale, rotation)
        normalized = normalize_landmarks(landmarks)
        
        # 2. Extract features (landmarks + hand-crafted features)
        features = extract_basic_features(normalized)
        
        # 3. Flatten and reshape for model (1, -1)
        features = features.reshape(1, -1)
        
        # 4. Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        label = self.labels[class_idx]
        return {"prediction": label, "confidence": confidence}, None
