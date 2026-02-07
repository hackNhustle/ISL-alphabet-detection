
import os
import sys
import numpy as np
import cv2

# Add the alphabet service app directory to path
service_dir = '/Users/emaad/Developer/emaad/isl_det/services/alphabet'
sys.path.append(service_dir)

from app.predictor import AlphabetPredictor

def test_predictor():
    print("Initializing AlphabetPredictor...")
    try:
        predictor = AlphabetPredictor()
        print("✓ Predictor initialized successfuly")
    except Exception as e:
        print(f"✗ Failed to initialize predictor: {e}")
        return

    # Create a dummy image (black image)
    # Alphabet predictor will fail if no hand is detected, so we'll check that error case first.
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', dummy_image)
    image_bytes = img_encoded.tobytes()

    print("\nTesting prediction with empty image (should return 'No hand detected')...")
    result, error = predictor.predict(image_bytes)
    
    if error == "No hand detected":
        print("✓ Correctly handled image with no hand")
    else:
        print(f"✗ Unexpected result: result={result}, error={error}")

    # To test a positive case, we would need an image with a hand landmarker.
    # Since we don't have one handy, we can at least verify that the scaler.transform 
    # call would work if we mock the hand extraction.
    
    print("\nVerifying scaler and preprocessing logic...")
    # Mock landmarks (21, 3)
    mock_landmarks = np.random.rand(21, 3)
    
    from app.preprocessing import normalize_landmarks, extract_basic_features
    
    try:
        normalized = normalize_landmarks(mock_landmarks)
        features = extract_basic_features(normalized)
        features = features.reshape(1, -1)
        
        # This is where it failed before
        scaled = predictor.scaler.transform(features)
        print("✓ Scaler transformation successful")
        
        # Test model prediction call (won't return meaningful result but shouldn't crash)
        model_result = predictor.model.predict(scaled, verbose=0)
        print("✓ Model prediction logical flow confirmed")
        
    except Exception as e:
        print(f"✗ Preprocessing/Scaling failed: {e}")

if __name__ == "__main__":
    test_predictor()
