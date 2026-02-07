"""
MediaPipe Tasks wrapper for consistent landmark extraction
Handles both IMAGES (alphabets) and VIDEOS (words)
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import os
import time

class PoseExtractor:
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=1,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 model_path='models/hand_landmarker.task'):
        """
        Initialize MediaPipe Hand Landmarker Task
        """
        if not os.path.exists(model_path):
            # Try to find it in the project root if called from app/
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"MediaPipe model not found at {model_path}. Please download it.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        
        self.running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.static_image_mode = static_image_mode
        self._frame_timestamp_ms = 0
        
    def extract_from_frame(self, frame, timestamp_ms=None):
        """
        Extract landmarks from single frame
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        if self.running_mode == vision.RunningMode.IMAGE:
            results = self.landmarker.detect(mp_image)
        else:
            # For video mode, we need a monotonically increasing timestamp
            if timestamp_ms is None:
                self._frame_timestamp_ms += 33 # Assume 30 FPS if not provided
                timestamp_ms = self._frame_timestamp_ms
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not results.hand_landmarks:
            return None, None
        
        # Get first hand
        hand_landmarks = results.hand_landmarks[0]
        # Handedness: results.handedness is a list of lists of Category objects
        handedness = results.handedness[0][0].category_name if results.handedness else None
        
        # Convert to numpy array (21, 3)
        landmarks = np.array([
            [lm.x, lm.y, lm.z] 
            for lm in hand_landmarks
        ])
        
        return landmarks, handedness
    
    def extract_from_video(self, video_path):
        """
        Extract landmark SEQUENCE from video file
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_ms_interval = int(1000 / fps)
        
        sequence = []
        handedness_votes = []
        
        frame_count = 0
        detected_count = 0
        current_timestamp_ms = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            landmarks, handedness = self.extract_from_frame(frame, timestamp_ms=current_timestamp_ms)
            current_timestamp_ms += frame_ms_interval
            
            sequence.append(landmarks)
            
            if landmarks is not None:
                handedness_votes.append(handedness)
                detected_count += 1
        
        cap.release()
        
        if detected_count == 0:
            return None, None
        
        # Interpolate missing frames
        sequence = self._interpolate_landmarks(sequence)
        sequence = [frame for frame in sequence if frame is not None]
        
        if not sequence:
            return None, None
        
        # Majority vote for handedness
        if handedness_votes:
            dominant_hand = max(set(handedness_votes), key=handedness_votes.count)
        else:
            dominant_hand = None
        
        return sequence, dominant_hand

    def _interpolate_landmarks(self, sequence):
        """
        Fill missing frames (None) using linear interpolation
        """
        seq_len = len(sequence)
        
        # Find indices of present frames
        present_indices = [i for i, frame in enumerate(sequence) if frame is not None]
        
        if not present_indices:
            return sequence
            
        # If we have gaps
        if len(present_indices) < seq_len:
            # We can only interpolate between first and last detected frame
            # (extrapolation is risky)
            first_idx = present_indices[0]
            last_idx = present_indices[-1]
            
            for i in range(first_idx, last_idx):
                if sequence[i] is None:
                    # Find previous and next valid frames
                    prev_idx = i - 1
                    while prev_idx >= 0 and sequence[prev_idx] is None:
                        prev_idx -= 1
                        
                    next_idx = i + 1
                    while next_idx < seq_len and sequence[next_idx] is None:
                        next_idx += 1
                        
                    if prev_idx >= 0 and next_idx < seq_len:
                        # Interpolate
                        prev_frame = sequence[prev_idx]
                        next_frame = sequence[next_idx]
                        
                        # Linear interpolation factor
                        alpha = (i - prev_idx) / (next_idx - prev_idx)
                        
                        interpolated_frame = prev_frame + alpha * (next_frame - prev_frame)
                        sequence[i] = interpolated_frame
                        
        return sequence
    
    def save_landmarks(self, landmarks, filepath, metadata=None):
        """
        Save landmarks to JSON
        
        Args:
            landmarks: Either (21, 3) array OR list of (21, 3) arrays
            filepath: Output JSON path
            metadata: Optional dict with extra info
        """
        # Handle both single frame and sequences
        if isinstance(landmarks, np.ndarray):
            if landmarks.ndim == 2:
                # Single frame: (21, 3)
                landmarks_list = landmarks.tolist()
            elif landmarks.ndim == 3:
                # Sequence: (T, 21, 3)
                landmarks_list = [frame.tolist() for frame in landmarks]
            else:
                raise ValueError(f"Unexpected landmarks shape: {landmarks.shape}")
        elif isinstance(landmarks, list):
            # Already a list (from video extraction)
            landmarks_list = [lm.tolist() if isinstance(lm, np.ndarray) else lm 
                             for lm in landmarks]
        else:
            raise TypeError(f"Unexpected landmarks type: {type(landmarks)}")
        
        # Check if it's a sequence based on nesting
        # Single frame: [[x,y,z], ...] -> Element is list of floats
        # Sequence: [[[x,y,z], ...], ...] -> Element is list of lists
        is_sequence = False
        if landmarks_list and isinstance(landmarks_list[0], list):
            if landmarks_list[0] and isinstance(landmarks_list[0][0], list):
                is_sequence = True
        
        data = {
            'landmarks': landmarks_list,
            'is_sequence': is_sequence,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_landmarks(self, filepath):
        """
        Load landmarks from JSON
        
        Returns:
            landmarks: Either (21, 3) array OR list of (21, 3) arrays
            metadata: Dict
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        landmarks_list = data['landmarks']
        is_sequence = data.get('is_sequence', False)
        
        if is_sequence:
            # Sequence of frames
            landmarks = [np.array(frame) for frame in landmarks_list]
        else:
            # Single frame
            landmarks = np.array(landmarks_list)
        
        return landmarks, data.get('metadata', {})
    
    def close(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass
