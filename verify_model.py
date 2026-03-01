import os
import sys
import random
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# Fix for potential Keras import issues
try:
    import keras
except ImportError:
    from tensorflow import keras

from utils.config import Paths, IMG_SIZE
from utils.image_ops import preprocess_image
from utils.labels import class_name, GTSRB_CLASSES

# Setup paths
project_root = Path(__file__).resolve().parent
paths = Paths(project_root=project_root)
model_path = paths.cnn_dir / "model.keras"

def verify_single_image(model, img_path, expected_class_id):
    if not img_path.exists():
        print(f"Skipping {img_path} (not found)")
        return

    # Read and preprocess
    # Use cv2.imread matching the app's behavior (BGR -> RGB)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Failed to read {img_path}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    img_preprocessed = preprocess_image(img_rgb, size=IMG_SIZE)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    # Predict
    probs = model.predict(img_batch, verbose=0)[0]
    pred_class = np.argmax(probs)
    confidence = np.max(probs)
    
    expected_name = class_name(expected_class_id)
    pred_name = class_name(pred_class)
    
    is_match = (pred_class == expected_class_id)
    status = "SUCCESS" if is_match else "FAILURE"
    
    print(f"[{status}] Image: {img_path.name}")
    print(f"  Expected: {expected_name} ({expected_class_id})")
    print(f"  Predicted: {pred_name} ({pred_class})")
    print(f"  Confidence: {confidence:.2%}")
    if not is_match:
        # Show confidence for the expected class
        expected_conf = probs[expected_class_id]
        print(f"  Confidence for correct class: {expected_conf:.2%}")
        
    print("-" * 30)
    return is_match

def main():
    # Redirect output to file manually since shell redirection is failing
    # Use a hardcoded absolute path to be 100% sure where it ends up
    log_file = Path(r"d:\traffic sign classifier\verify_log.txt")
    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    
    if not model_path.exists():
        print(f"Model not found at {model_path}!")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Class 2: Speed limit (50km/h)
    # Class 25: Road work
    # We will pick a few images from train/Final_Training/Images/00002 and 00025
    
    train_images_root = paths.dataset_root / "train/Final_Training/Images"
    
    # Test all 43 classes
    classes_to_test = list(range(43))
    
    total_passed = 0
    total_tested = 0
    
    print(f"Starting verification for {len(classes_to_test)} classes...")
    
    for cls_id in classes_to_test:
        cls_folder = train_images_root / f"{cls_id:05d}"
        if not cls_folder.exists():
            print(f"Class folder {cls_folder} not found. Skipping class {cls_id}.")
            continue
            
        print(f"\nTesting Class {cls_id}: {class_name(cls_id)}")
        
        # Get all images
        images = list(cls_folder.glob("*.jpg"))
        # Also check for png/jpg just in case
        if not images:
            images = list(cls_folder.glob("*.png")) + list(cls_folder.glob("*.jpg"))
            
        if not images:
            print("No images found in this class folder.")
            continue
            
        # Pick 3 random images
        sample_images = random.sample(images, min(3, len(images)))
        
        for img_path in sample_images:
            if verify_single_image(model, img_path, cls_id):
                total_passed += 1
            total_tested += 1
            
    print(f"\nTotal Result: {total_passed}/{total_tested} Passed")

if __name__ == "__main__":
    main()
