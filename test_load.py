import sys
import traceback

print("Testing Model Load Sequence...")

try:
    print("Attempting to import building functions...")
    from cnn.model import build_cnn
    from utils.config import NUM_CLASSES, IMG_SIZE
    
    print("Building blank model architecture...")
    model = build_cnn(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    
    print("Testing weight loaded from Keras 3 format...")
    model.load_weights("artifacts/cnn/model.keras")
    
    print("Saving to H5 format for testing...")
    model.save("artifacts/cnn/test_model.h5")
    
    print("SUCCESS")
except Exception as e:
    print(f"FAILED. Exception type: {type(e)}")
    traceback.print_exc()
