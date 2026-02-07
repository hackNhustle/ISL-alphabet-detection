"""
Normalization and preprocessing functions
CRITICAL: These must be identical for training and inference
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def normalize_landmarks(landmarks):
    """
    Normalize single frame landmarks
    
    Input: (21, 3) array
    Output: (21, 3) normalized array
    
    Steps:
    1. Translate wrist to origin
    2. Scale to unit hand size
    3. Rotate to canonical orientation
    """
    landmarks = landmarks.copy()
    
    # 1. Center on wrist (landmark 0)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    
    # 2. Scale normalization
    # Use distance from wrist to middle finger base as reference
    reference_distance = np.linalg.norm(landmarks[9])  # middle finger base
    
    if reference_distance > 0:
        landmarks = landmarks / reference_distance
    
    # 3. Rotation normalization (align to x-axis)
    # Vector from wrist to middle finger base
    reference_vector = landmarks[9][:2]  # x, y only
    
    if np.linalg.norm(reference_vector) > 0:
        angle = np.arctan2(reference_vector[1], reference_vector[0])
        
        # Rotation matrix (2D)
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Apply rotation to x, y coordinates
        landmarks[:, :2] = landmarks[:, :2] @ rotation_matrix.T
    
    return landmarks

def flatten_landmarks(landmarks):
    """
    (21, 3) -> (63,)
    """
    return landmarks.flatten()

def extract_basic_features(landmarks):
    """
    Extract hand-crafted features from landmarks
    
    Returns: feature vector
    """
    features = []
    
    # 1. Flattened normalized landmarks
    features.append(flatten_landmarks(landmarks))
    
    # 2. Inter-finger distances
    finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    for i in range(len(finger_tips)):
        for j in range(i + 1, len(finger_tips)):
            dist = np.linalg.norm(
                landmarks[finger_tips[i]] - landmarks[finger_tips[j]]
            )
            features.append([dist])
    
    # 3. Finger curl (distance from tip to base)
    finger_bases = [2, 5, 9, 13, 17]
    for tip, base in zip(finger_tips, finger_bases):
        curl = np.linalg.norm(landmarks[tip] - landmarks[base])
        features.append([curl])
    
    # 4. Palm center
    palm_landmarks = [0, 5, 9, 13, 17]  # wrist + finger bases
    palm_center = np.mean(landmarks[palm_landmarks], axis=0)
    features.append(palm_center)
    
    return np.concatenate(features)

class LandmarkPreprocessor:
    """
    Stateful preprocessor that can be saved/loaded
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X):
        """
        Fit scaler on training data
        X: (n_samples, n_features)
        """
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply normalization"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        """Fit and transform"""
        return self.fit(X).transform(X)
    
    def save(self, filepath):
        """Save preprocessor state"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.scaler = data['scaler']
        preprocessor.is_fitted = data['is_fitted']
        return preprocessor
