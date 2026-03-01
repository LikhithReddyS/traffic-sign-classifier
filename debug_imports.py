import sys
import subprocess

with open("debug_log.txt", "w") as f:
    orig_stdout = sys.stdout
    sys.stdout = f
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        try:
            print(f"tf.keras version: {tf.keras.__version__}")
        except AttributeError:
            print("tf.keras is NOT available")
        except Exception as e:
            print(f"Error accessing tf.keras: {e}")

        try:
            import keras
            print(f"Keras version: {keras.__version__}")
        except ImportError:
            print("Keras package not found")
    except ImportError:
        print("TensorFlow not found")
    except Exception as e:
        import traceback
        traceback.print_exc()

    print("Package list:")
    subprocess.run([sys.executable, "-m", "pip", "list"], stdout=f, stderr=subprocess.STDOUT)
    
    sys.stdout = orig_stdout
