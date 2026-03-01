import sys
from pathlib import Path
import importlib

def check_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"[OK] {module_name} imported.")
        return True
    except ImportError as e:
        print(f"[FAIL] {module_name} could not be imported: {e}")
        return False

def main():
    print("Checking Setup...")
    
    # 1. Check Dependencies
    dependencies = [
        "numpy",
        "pandas",
        "sklearn", # scikit-learn
        "tensorflow",
        "cv2", # opencv-python
        "streamlit",
        "tqdm",
        "PIL" # Pillow
    ]
    
    all_deps_ok = True
    for dep in dependencies:
        if not check_import(dep):
            all_deps_ok = False
        else:
            # Print version if available
            try:
                mod = importlib.import_module(dep)
                version = getattr(mod, "__version__", "unknown")
                print(f"  - {dep} version: {version}")
            except Exception:
                pass

    if all_deps_ok:
        print("\nAll dependencies installed.")
        # Extra check for tf.keras
        try:
            import tensorflow as tf
            if not hasattr(tf, "keras"):
                print("[FAIL] tensorflow.keras is missing! (Common in TF 2.16+ without tf_keras)")
                print("Suggestion: pip install tensorflow==2.15.0")
            else:
                print(f"[OK] tf.keras is available (TF {tf.__version__})")
        except ImportError:
            pass
    else:
        print("\nMissing dependencies. Please run: pip install -r requirements.txt")

    # 2. Check Dataset
    project_root = Path(__file__).resolve().parent
    dataset_root = project_root / "dataset"
    train_root = dataset_root / "train" / "Final_Training" / "Images"
    test_root = dataset_root / "test" / "Final_Test" / "Images"
    test_csv_a = dataset_root / "GT-final_test.csv"
    test_csv_b = test_root / "GT-final_test.csv"

    if train_root.exists() and test_root.exists() and (test_csv_a.exists() or test_csv_b.exists()):
        print(f"[OK] Dataset directory found at {dataset_root}")
        print(f"     Train images: {train_root}")
        print(f"     Test images:  {test_root}")
    else:
        print(f"[FAIL] Dataset layout not found under {dataset_root}")
        print("Expected:")
        print(f"  - {train_root}")
        print(f"  - {test_root}")
        print(f"  - {test_csv_a} (or {test_csv_b})")
        print("Please download GTSRB and extract it accordingly.")
         
    # 3. Check GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] GPU Detected: {len(gpus)} device(s).")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("[INFO] No GPU detected. Training will be slow.")
    except Exception as e:
        print(f"[WARN] Could not check GPU: {e}")

if __name__ == "__main__":
    main()
