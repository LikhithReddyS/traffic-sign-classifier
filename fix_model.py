from pathlib import Path

"""Artifact compatibility helper.

This project is often trained in an environment with newer Keras (e.g. Keras 3)
but run locally with TF/Keras 2.x. Full-model formats (`model.keras`, some `model.h5`)
can fail to deserialize across those versions.

The most portable handoff format is a **weights-only** file:
`artifacts/cnn/model.weights.h5`

Generate it in the same environment that can load `model.keras`, then copy it to
this repo and the Streamlit app will prefer it automatically.
"""

def convert_model():
    model_path = Path("artifacts/cnn/model.keras")
    h5_path = Path("artifacts/cnn/model.h5")
    weights_path = Path("artifacts/cnn/model.weights.h5")
    
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        return

    print(
        "This script can only convert if your current Python environment can load `model.keras`.\n"
        "If you're on TF/Keras 2.x and `model.keras` was produced by Keras 3, conversion must be done\n"
        "in the original training environment.\n\n"
        f"Expected inputs/outputs:\n- input: {model_path}\n- output: {weights_path} (preferred) or {h5_path}\n"
    )

    # Keep the script as a placeholder with clear guidance rather than failing
    # with confusing cross-version deserialization errors.
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        return

    print(
        "Suggested fix (run where training happened, e.g. Colab):\n"
        "  model = keras.models.load_model('model.keras', compile=False)\n"
        "  model.save_weights('model.weights.h5')\n"
        "Then copy `model.weights.h5` into `artifacts/cnn/` locally."
    )

if __name__ == "__main__":
    convert_model()
