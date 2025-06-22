import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def analyze_image_features(image_path):
    """Analyze basic image features to help identify potential fake images"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Error: Could not load image"
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Basic analysis
        height, width, channels = img.shape
        
        # Calculate basic statistics
        mean_color = np.mean(img_rgb, axis=(0, 1))
        std_color = np.std(img_rgb, axis=(0, 1))
        
        # Edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Texture analysis (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Color histogram analysis
        hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
        
        # Calculate histogram statistics
        hist_std_r = np.std(hist_r)
        hist_std_g = np.std(hist_g)
        hist_std_b = np.std(hist_b)
        
        # Simple heuristic for fake detection
        # Fake images often have:
        # 1. Unusual color distributions
        # 2. Artificial-looking edges
        # 3. Inconsistent texture patterns
        
        fake_score = 0
        
        # Check for unusual color distributions
        if np.max(mean_color) - np.min(mean_color) > 50:
            fake_score += 0.2
        
        # Check for high edge density (artificial sharpening)
        if edge_density > 0.1:
            fake_score += 0.3
        
        # Check for low texture variance (smooth artificial areas)
        if laplacian_var < 100:
            fake_score += 0.2
        
        # Check for unusual histogram patterns
        if hist_std_r < 1000 or hist_std_g < 1000 or hist_std_b < 1000:
            fake_score += 0.3
        
        # Normalize score
        fake_score = min(fake_score, 1.0)
        
        # Determine classification
        if fake_score > 0.5:
            classification = "Fake"
            confidence = fake_score
        else:
            classification = "Real"
            confidence = 1 - fake_score
        
        return {
            'classification': classification,
            'confidence': confidence,
            'fake_score': fake_score,
            'image_size': (width, height),
            'mean_color': mean_color,
            'edge_density': edge_density,
            'texture_variance': laplacian_var,
            'histogram_std': (hist_std_r, hist_std_g, hist_std_b)
        }, None
        
    except Exception as e:
        return None, f"Error analyzing image: {str(e)}"

def display_analysis(image_path, analysis):
    """Display the image analysis results"""
    if analysis is None:
        print("No analysis available")
        return
    
    print(f"\n=== Image Analysis Results ===")
    print(f"Classification: {analysis['classification']}")
    print(f"Confidence: {analysis['confidence']:.2%}")
    print(f"Fake Score: {analysis['fake_score']:.3f}")
    print(f"Image Size: {analysis['image_size']}")
    print(f"Mean Color (RGB): {analysis['mean_color']}")
    print(f"Edge Density: {analysis['edge_density']:.4f}")
    print(f"Texture Variance: {analysis['texture_variance']:.2f}")
    print(f"Histogram Std (RGB): {analysis['histogram_std']}")

def main():
    print("=== DeepGuard: Simple Image Analysis ===")
    print("Analyzing images using basic computer vision techniques")
    print("Note: This is a simplified analysis, not a trained deep learning model")
    print()
    
    # Test specific images
    test_images = [
        (r"F:\DeepGuard\Data\Fake\fake_10.jpg", "Fake Image (fake_10.jpg)"),
        (r"F:\DeepGuard\Data\Real\real_1006.jpg", "Real Image (real_1006.jpg)")
    ]
    
    for image_path, description in test_images:
        print(f"\n{'='*50}")
        print(f"Testing: {description}")
        print(f"Path: {image_path}")
        
        if os.path.exists(image_path):
            analysis, error = analyze_image_features(image_path)
            if error:
                print(f"Error: {error}")
            else:
                display_analysis(image_path, analysis)
        else:
            print(f"Error: Image not found at {image_path}")
    
    print(f"\n{'='*50}")
    print("Analysis Complete!")
    print("Note: This analysis uses basic image processing techniques.")
    print("For more accurate results, a properly trained deep learning model is recommended.")

if __name__ == "__main__":
    main() 