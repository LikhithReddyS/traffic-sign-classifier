from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.config import IMG_SIZE, Paths
from utils.image_ops import preprocess_frame_for_model, preprocess_image
from utils.labels import class_name


# Handle Keras import for different TF versions (2.x vs Keras 3)
try:
    import keras
except ImportError:
    from tensorflow import keras


# ...

@st.cache_resource
def load_model_cached(model_path: str, model_mtime: float):
    """Load a model in a way that survives Keras/TF version differences.

    Preference order:
    1) Load the full model (works for `.keras` and many `.h5` saves).
    2) Fall back to rebuilding the architecture from source and loading weights.

    `model_mtime` is only used to bust Streamlit cache when the file changes.
    """
    from cnn.model import build_cnn, build_cnn_legacy
    from utils.config import NUM_CLASSES, IMG_SIZE

    expected_input_shape = (None, IMG_SIZE, IMG_SIZE, 3)
    expected_output_shape = (None, NUM_CLASSES)

    # Attempt 1: load the full serialized model.
    try:
        model = keras.models.load_model(model_path, compile=False)
        if tuple(model.input_shape) != expected_input_shape or tuple(model.output_shape) != expected_output_shape:
            raise ValueError(
                "Loaded model has unexpected shapes: "
                f"input={model.input_shape}, output={model.output_shape}. "
                f"Expected input={expected_input_shape}, output={expected_output_shape}."
            )
        return model
    except Exception:
        pass

    # Attempt 2: rebuild and load weights only.
    # First try the current architecture; if it doesn't match the saved weights,
    # fall back to the legacy architecture (no BatchNorm).
    for builder in (build_cnn, build_cnn_legacy):
        model = builder(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
        try:
            model.load_weights(model_path)
            return model
        except ValueError:
            continue

    raise ValueError(
        "Failed to load model weights. The saved artifact does not match either the "
        "current or legacy CNN architecture. Re-run training/conversion to regenerate "
        "artifacts (e.g. `python -m cnn.train_cnn`) and ensure the app points to the "
        "matching file in `artifacts/cnn/`."
    )




def _predict(model: tf.keras.Model, img_rgb_01: np.ndarray):
    """Run prediction on a preprocessed image and return (class_id, confidence, probs)."""
    probs = model.predict(np.expand_dims(img_rgb_01, 0), verbose=0)[0]
    cls = int(np.argmax(probs))
    conf = float(np.max(probs))
    return cls, conf, probs


def _draw_overlay(frame_bgr: np.ndarray, cls: int, conf: float) -> np.ndarray:
    """Draw prediction overlay on the frame with bounding box and label."""
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    # Draw a guide bounding box in the center region of the frame
    box_margin_x = int(w * 0.15)
    box_margin_y = int(h * 0.10)
    x1, y1 = box_margin_x, box_margin_y
    x2, y2 = w - box_margin_x, h - box_margin_y

    # Bounding box color: green if confident, yellow if medium, red if low
    if conf >= 0.7:
        color = (0, 255, 0)  # Green
    elif conf >= 0.4:
        color = (0, 200, 255)  # Yellow-orange
    else:
        color = (0, 0, 255)  # Red

    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

    # Prediction label text
    label = f"{class_name(cls)}  {conf:.2f}"

    # Calculate text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw filled rectangle behind text
    text_x = x1
    text_y = y1 - 10
    cv2.rectangle(
        overlay,
        (text_x, text_y - text_h - baseline - 5),
        (text_x + text_w + 10, text_y + 5),
        color,
        cv2.FILLED,
    )

    # Draw text
    cv2.putText(
        overlay,
        label,
        (text_x + 5, text_y - 5),
        font,
        font_scale,
        (255, 255, 255),  # White text
        thickness,
        cv2.LINE_AA,
    )

    return overlay


def _display_prediction_box(cls: int, conf: float):
    """Display a styled prediction result box below the image."""
    name = class_name(cls)
    
    # Determine style based on confidence
    if conf >= 0.7:
        st.success(f"**Prediction:** {name}  \n**Confidence:** {conf:.2%}")
    elif conf >= 0.4:
        st.warning(f"**Prediction:** {name}  \n**Confidence:** {conf:.2%} (Uncertain)")
    else:
        st.error(f"**Prediction:** {name}  \n**Confidence:** {conf:.2%} (Low Confidence)")


def _get_camera():
    """Get or create a persistent camera capture object via session state."""
    if "camera_cap" not in st.session_state or st.session_state.camera_cap is None:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            st.session_state.camera_cap = cap
        else:
            return None
    return st.session_state.camera_cap


def _release_camera():
    """Release the persistent camera capture object."""
    if "camera_cap" in st.session_state and st.session_state.camera_cap is not None:
        try:
            st.session_state.camera_cap.release()
        except Exception:
            pass
        st.session_state.camera_cap = None


def main() -> None:
    st.set_page_config(page_title="Traffic Sign Recognition", layout="wide")

    st.title("Traffic Sign Recognition System")
    st.caption(
        "CNN Traffic Sign Recognition System · "
        "Semantics-preserving augmentation (no rotation/flipping)"
    )

    # Inject custom CSS for image borders
    st.markdown(
        """
        <style>
        img {
            border: 5px solid #444;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    project_root = Path(__file__).resolve().parent
    paths = Paths(project_root=project_root)

    # Preferred artifact order:
    # 1) `model.weights.h5` (weights-only; best cross-version portability)
    # 2) `model.h5`         (legacy weights-only in this repo)
    # 3) `model.keras`      (full model; may be produced by newer Keras)
    model_path = paths.cnn_dir / "model.weights.h5"
    if not model_path.exists():
        model_path = paths.cnn_dir / "model.h5"
    if not model_path.exists():
        model_path = paths.cnn_dir / "model.keras"
    metrics_path = paths.cnn_dir / "metrics.json"

    # ── Model loading ──────────────────────────────────────────────
    if not model_path.exists():
        st.error(
            "Model not found in artifacts. Train the CNN first "
            "(`python -m cnn.train_cnn`) to generate `artifacts/cnn/model.keras`."
        )
        st.stop()

    model = load_model_cached(str(model_path), model_path.stat().st_mtime)

    # ── Sidebar: Offline metrics ───────────────────────────────────
    with st.sidebar:
        st.subheader("📊 Offline Metrics")
        st.caption(f"Model artifact: {model_path.name}")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.4f}")
            st.metric("Precision (macro)", f"{metrics.get('precision_macro', 0.0):.4f}")
            st.metric("Precision (weighted)", f"{metrics.get('precision_weighted', 0.0):.4f}")
        else:
            st.info("Run `python -m cnn.evaluate` to generate metrics.")

    # ── Input mode selection ───────────────────────────────────────
    input_mode = st.radio(
        "Choose input mode",
        ["📷 Camera (Browser)", "🖼️ Upload Image", "🎥 Camera (OpenCV)"],
        horizontal=True,
    )

    if input_mode != "🎥 Camera (OpenCV)":
        _release_camera()

    st.divider()

    # ── MODE 1: Browser Camera ─────────────────────────────────────
    if input_mode == "📷 Camera (Browser)":
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Capture from Camera")
            st.info("Place the traffic sign in the CENTER of the frame.")
            camera_photo = st.camera_input("Take a photo of a traffic sign")

        with col2:
            st.subheader("Prediction")
            if camera_photo is not None:
                pil_img = Image.open(camera_photo).convert("RGB")
                img_np = np.array(pil_img)
                
                # Center crop for better accuracy
                h, w = img_np.shape[:2]
                min_dim = min(h, w)
                start_x = (w - min_dim) // 2
                start_y = (h - min_dim) // 2
                img_cropped = img_np[start_y:start_y+min_dim, start_x:start_x+min_dim]

                img_processed = preprocess_image(img_cropped, size=IMG_SIZE)
                cls, conf, _ = _predict(model, img_processed)

                # Draw overlay on the cropped image
                img_bgr = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)
                img_overlay = _draw_overlay(img_bgr, cls, conf)
                img_overlay_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

                st.image(img_overlay_rgb, caption="Detection result (Center Cropped)", width="stretch")
                _display_prediction_box(cls, conf)
            else:
                st.info("Click the camera button to capture a traffic sign.")

    # ── MODE 2: Image Upload ───────────────────────────────────────
    elif input_mode == "🖼️ Upload Image":
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🖼️ Upload a Traffic Sign Image")
            uploaded = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png", "bmp"],
            )

        with col2:
            st.subheader("Prediction")
            if uploaded is not None:
                pil_img = Image.open(uploaded).convert("RGB")
                img_np = np.array(pil_img)

                img_processed = preprocess_image(img_np, size=IMG_SIZE)
                cls, conf, _ = _predict(model, img_processed)

                # Draw overlay on the uploaded image
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img_overlay = _draw_overlay(img_bgr, cls, conf)
                img_overlay_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

                st.image(img_overlay_rgb, caption="Detection result", width="stretch")
                _display_prediction_box(cls, conf)
            else:
                st.info("Upload a traffic sign image to classify it.")

    # ── MODE 3: OpenCV Camera (continuous live feed) ───────────────
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🎥 OpenCV Camera Controls")

            if "run_camera" not in st.session_state:
                st.session_state.run_camera = False

            start = st.button("Start camera", type="primary", disabled=st.session_state.run_camera)
            stop = st.button("Stop camera", disabled=not st.session_state.run_camera)

            if start:
                st.session_state.run_camera = True
            if stop:
                st.session_state.run_camera = False
                _release_camera()

            st.caption("💡 Place sign in the **BLUE BOX** for best accuracy.")

        with col2:
            st.subheader("Live Feed")
            frame_placeholder = st.empty()
            
            # Placeholder for prediction box (so it updates in place)
            prediction_placeholder = st.empty()

            if st.session_state.run_camera:
                cap = _get_camera()

                if cap is None or not cap.isOpened():
                    st.error(
                        "Could not open camera.\n\n"
                        "**Switch to 📷 Camera (Browser) mode.**"
                    )
                    st.session_state.run_camera = False
                    _release_camera()
                    st.stop()

                try:
                    while st.session_state.run_camera:
                        ok, frame = cap.read()
                        if not ok:
                            st.warning("Failed to read frame.")
                            break
                        
                        # Define ROI (Central Square)
                        h, w = frame.shape[:2]
                        box_size = min(h, w) // 2  # Use half the screen size for the box
                        start_x = (w - box_size) // 2
                        start_y = (h - box_size) // 2
                        end_x = start_x + box_size
                        end_y = start_y + box_size
                        
                        # Crop for prediction
                        roi = frame[start_y:end_y, start_x:end_x]

                        x = preprocess_frame_for_model(roi, size=IMG_SIZE)
                        cls, conf, _ = _predict(model, x)

                        # Draw overlay on the FULL frame
                        # 1. Draw the Blue Guide Box
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
                        
                        # 2. Draw prediction on the ROI part of the frame (optional, or just label)
                        # We will draw the result label above the blue box
                        label = f"{class_name(cls)} {conf:.2f}"
                        color = (0, 255, 0) if conf > 0.7 else (0, 0, 255)
                        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")
                        
                        # Update prediction box
                        with prediction_placeholder.container():
                            _display_prediction_box(cls, conf)
                            
                except Exception:
                    _release_camera()
            else:
                st.info("Click 'Start camera' to begin real-time recognition.")


if __name__ == "__main__":
    main()
