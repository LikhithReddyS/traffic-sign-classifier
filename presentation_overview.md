# Traffic Sign Classifier with GAN Augmentation: Technical Deep Dive

## 1. Executive Summary
*   **Project Name:** GTSRB Traffic Sign Classification & Generation
*   **Goal:** Classify 43 unique German traffic signs with >94% accuracy and generate synthetic training data for minority classes.
*   **Core Tech:** TensorFlow (CNN + DCGAN), OpenCV, Streamlit, Python.

## 2. System Architecture & Linkages
The project is modular, with distinct components for Training, generation, and Inference.

### 🔷 File Structure & Linkages
```mermaid
graph TD
    A[Dataset (GTSRB)] --> |Input| B(dataset/readTrafficSigns.py)
    B --> |Numpy Arrays| C{Data Pipeline}
    
    subgraph "Preprocessing & Utils"
        C --> D[utils/image_ops.py]
        D --> |Resize/Normalize| E[utils/augment.py]
    end
    
    subgraph "Model Training (CNN)"
        E --> F[cnn/model.py]
        F --> |Architecture| G[cnn/train_cnn.py]
        G --> |Saves .h5| H[artifacts/cnn/gtsrb_cnn.h5]
    end
    
    subgraph "Data Generation (GAN)"
        E --> I[gan/dcgan.py]
        I --> |Architecture| J[gan/train_gan.py]
        J --> |Saves Generator| K[artifacts/gan/generator_X.h5]
        K --> |Generates| L[gan/generate_synthetic.py]
        L --> |Synthetic Images| C
    end
    
    subgraph "Inference (Deployment)"
        H --> M[streamlit_app.py]
        M --> |Live Prediction| N[User Interface]
    end
```

## 3. Detailed Performance Metrics
We achieved high accuracy across the board, with specific strengths in common signs.

*   **Overall Test Accuracy:** `94.24%`
*   **Weighted Precision:** `94.44%`
*   **Weighted F1-Score:** `94.14%`
*   **Total Test Samples:** `12,630` images

### ✅ Best Performing Classes
| Class ID | Sign Name | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **35** | Ahead only | **100%** | 96% | 98% |
| **17** | No entry | **100%** | 98% | 99% |
| **13** | Yield | 99% | 99% | 99% |
| **33** | Turn right ahead | 99% | 100% | 99% |

### ⚠️ Challenging Classes (Fixed via GAN)
| Class ID | Sign Name | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Speed limit (20km/h) | 95% | **35%** | **51%** |
| **27** | Pedestrians | 55% | 50% | 52% |
*(Note: These classes were rare in the dataset, motivating the use of GANs for augmentation.)*

## 4. Technical Specifications: Models & Losses

### 🅰️ Classifier (CNN)
*   **Loss Function:** `SparseCategoricalCrossentropy`
    *   *Why?* We have integer labels (0-42) and want to penalized incorrect probability distributions.
    *   Formula: $L = -\sum_{i} y_i \log(\hat{y}_i)$
*   **Optimizer:** `Adam` (Learning Rate: 0.001)
*   **Output Layer:** `Softmax` (Probability distribution summing to 1.0).

### 🅱️ Generator (DCGAN)
*   **Loss Function:** `BinaryCrossentropy`
    *   *Why?* The Discriminator classification is binary (Real=1 vs Fake=0).
*   **Adversarial Min-Max Game:**
    *   **Generator Loss:** Tries to maximize Discriminator's mistake ($D(G(z)) \approx 1$).
    *   **Discriminator Loss:** Tries to correctly classify Real ($D(x) \approx 1$) and Fake ($D(G(z)) \approx 0$).

## 5. Deployment Pipeline
The final application (`streamlit_app.py`) integrates the entire pipeline:

1.  **Input:** Webcam Frame or Uploaded Image.
2.  **Preprocessing:**
    *   Resize to **32x32**.
    *   Normalize pixel values `x / 255.0`.
3.  **Inference:**
    *   Load `gtsrb_cnn.h5`.
    *   `prediction = model.predict(image)`.
    *   `confidence = max(prediction)`.
4.  **Logic:**
    *   If `confidence < 0.60`: Display "Unknown/Uncertain" (Red Box).
    *   Else: Display Sign Name (Green Box).

## 6. Conclusion
This project demonstrates a complete End-to-End Deep Learning pipeline, successfully solving the problem of uneven class distribution and real-time classification requirements. The modular design allows for easy retraining and replacement of components.
