# Traffic Sign Recognition System using GAN-based Data Augmentation and CNN (Streamlit Real-Time)

## Problem definition

Traffic sign recognition is a core perception task for ADAS and autonomous driving. This project builds an end-to-end pipeline to classify **43 GTSRB classes** in real time from a live camera feed.

## Key constraint: semantics-preserving augmentation

Traffic signs are **direction- and orientation-sensitive** (e.g., left vs right arrows). Therefore this project **does not use**:

- Rotation
- Horizontal flip
- Vertical flip

Allowed preprocessing / augmentation:

- Resize to 32×32
- RGB conversion
- Normalization to [0, 1]
- Brightness / contrast adjustments
- Noise handling
- GAN-based synthetic generation (used only for training)

## Why GAN augmentation (vs. traditional augmentation)

Classic geometric augmentation (rotate/flip) can break semantics for traffic signs. A GAN can learn the data distribution of a class and generate additional samples **without imposing label-breaking transforms**. In this project, the GAN is trained mainly on minority classes to reduce imbalance.

> Future scope: a **Conditional GAN (cGAN)** can generate class-conditional samples with a single model. This repo implements a per-class DCGAN for simplicity and strict separation of GAN and CNN training.

## System architecture (diagram description)

1. **Data pipeline** loads GTSRB images → RGB → resize 32×32 → normalize.
2. **Minority class detection** identifies underrepresented classes.
3. **GAN module (DCGAN)** trains on a selected minority class → generates synthetic images stored under `dataset/GTSRB/synthetic/`.
4. **CNN classifier** trains on real + synthetic images.
5. **Evaluation** computes accuracy, macro precision, weighted precision, and confusion matrix.
6. **Streamlit app** runs real-time camera inference using OpenCV and the trained CNN.

## Repository structure

- `dataset/` – place the downloaded GTSRB dataset here
- `gan/` – DCGAN training + synthetic generation scripts
- `cnn/` – CNN model, training, evaluation
- `utils/` – preprocessing, augmentation (semantics-safe), dataset loading
- `streamlit_app.py` – real-time demo UI
- `requirements.txt`

## Dataset setup (GTSRB)

1. Download GTSRB from: https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
2. Extract into this repo's expected layout:
   - `dataset/train/Final_Training/Images/...`
   - `dataset/test/Final_Test/Images/...`

The loader expects the official CSV files:
- `Final_Training/Images/00000/GT-00000.csv` (and similarly for other classes)
- `GT-final_test.csv` (either at `dataset/GT-final_test.csv` or at `dataset/test/Final_Test/Images/GT-final_test.csv`)

## Training workflow
### 1) Train GAN for a minority class

Train a DCGAN on a specific class (repeat for multiple minority classes if needed):

```bash
python -m gan.train_gan --class_id 41 --epochs 50
```

This saves a generator to `artifacts/gan/class_41/generator.keras`.

### 2) Generate synthetic images (training-only)

```bash
python -m gan.generate_synthetic --class_id 41 --n 2000
```

Synthetic images are written to `dataset/synthetic/class_41/`.

### 3) Train CNN classifier (real + synthetic)

```bash
python -m cnn.train_cnn --epochs 20
```

Model artifact saved to `artifacts/cnn/model.keras`.

### 4) Evaluate

```bash
python -m cnn.evaluate
```

Writes metrics to `artifacts/cnn/metrics.json` (accuracy, precision macro/weighted, confusion matrix).

## Real-time inference pipeline

1. OpenCV reads frames from a camera (`cv2.VideoCapture`).
2. Frame preprocessing matches training:
   - BGR → RGB
   - resize 32×32
   - normalize [0, 1]
3. CNN predicts probabilities; UI displays:
   - predicted class id
   - confidence score

Run the demo:

```bash
streamlit run streamlit_app.py
```

## Real-world automotive deployment notes

- **Integration**: The model would run behind the vehicle camera pipeline (GMSL/USB/MIPI camera → ISP → frame buffer → DNN inference).
- **Edge inference**: Suitable targets include NVIDIA Jetson (CUDA/TensorRT), Raspberry Pi (optimized TFLite), or automotive SoCs.
- **Latency**: Real-time perception typically requires low end-to-end latency (capture + preprocess + infer + postprocess). Use batching=1, optimized runtime (TFLite/TensorRT), and avoid heavy UI overhead.
- **Safety & reliability**:
  - Monitor confidence and add rejection/"unknown" handling for low-confidence predictions.
  - Validate in diverse conditions (night, rain, glare) and use calibration/monitoring.
  - Implement robust OOD detection and logging in production.

## Limitations

- The DCGAN is trained **per class** in this repo (simplifies minority-class focus but is slower than a single conditional model).
- Synthetic images may introduce artifacts; always validate that GAN samples improve classifier performance.
- Real-time demo assumes the sign is prominent in the camera frame (no detection/localization stage included).

## Future scope

- Conditional GAN (cGAN) or diffusion models for class-conditional synthesis.
- Add a detection stage (e.g., YOLO) to localize signs before classification.
- Export to TFLite / TensorRT, quantize for edge.
- Add class-name mapping for official GTSRB labels.
"# traffic-sign-classifier" 
"# traffic-sign-classifier" 
"# traffic-sign-classifier" 
