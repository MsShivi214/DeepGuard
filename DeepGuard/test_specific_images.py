import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load class names from labels.txt
LABELS_PATH = "converted_keras (1)/labels.txt"
with open(LABELS_PATH, "r") as f:
    lines = f.readlines()
    class_names = [line.strip().split(' ', 1)[1] for line in lines if line.strip()]

# Load the trained Keras model
MODEL_PATH = "converted_keras (1)/keras_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (224, 224)

def predict_image(image_path):
    """Predict class for a single image"""
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    pred = model.predict(img)[0][0]
    class_idx = int(pred > 0.5)
    class_name = class_names[class_idx]
    confidence = pred if pred > 0.5 else 1 - pred
    
    return class_name, float(confidence), float(pred)

def main():
    print("=== DeepGuard: Testing Specific Images ===")
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Labels loaded from: {LABELS_PATH}")
    print(f"Class names: {class_names}")
    print()
    
    # Test specific images
    test_images = [
        (r"F:\DeepGuard\Data\Fake\fake_10.jpg", "Fake Image"),
        (r"F:\DeepGuard\Data\Real\real_1006.jpg", "Real Image")
    ]
    
    for image_path, description in test_images:
        print(f"Testing {description}:")
        print(f"Path: {image_path}")
        
        if os.path.exists(image_path):
            class_name, confidence, raw_score = predict_image(image_path)
            if class_name is not None:
                print(f"  Prediction: {class_name}")
                print(f"  Confidence: {confidence:.2%}")
                print(f"  Raw Score: {raw_score:.4f}")
            else:
                print("  Error: Could not process image")
        else:
            print(f"  Error: Image not found at {image_path}")
        
        print()

if __name__ == "__main__":
    main() 