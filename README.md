# Real-Time Attentiveness Detection in Desk-Based Environments Using Facial Landmark and Deep Learning Features

Real-time, webcam-based student/worker attentiveness and engagement monitoring system that fuses **classical facial-landmark features** (via MediaPipe + a Random Forest / XGBoost / LightGBM / SVM-based hybrid classifier) with a **deep learning CNN (EfficientNet)** trained directly on cropped face images, to classify a subject's behavioral state into six fine-grained categories and two high-level engagement states.

> Built as part of ongoing research into affect-aware monitoring for desk-based environments (classrooms, home study, remote work), using only a standard RGB webcam — no specialized hardware required.

---

## 🎯 What it does

The system classifies the observed face into one of six behavioral states:

| Class | Engagement Group |
|---|---|
| Focused | Engaged |
| Frustrated | Engaged |
| Confused | Engaged |
| Looking Away | Non-Engaged |
| Drowsy | Non-Engaged |
| Bored | Non-Engaged |

It runs entirely in real time on a live webcam feed and:
- Detects the face and extracts 68/478-point facial landmarks using **MediaPipe Face Landmarker** and **Face Detector**.
- Computes geometric/behavioral features (eye aspect ratio, head tilt, brow position, mouth openness, gaze offset, etc.) and classifies them with a trained **Random Forest**.
- Crops and feeds the detected face into a fine-tuned **EfficientNet CNN** for an independent image-based prediction.
- **Fuses both predictions** with a weighted ensemble (35% landmark-based / 65% CNN-based) for a more robust final label.
- Applies temporal smoothing (rolling majority vote) to reduce flicker between frames.
- Tracks a running "non-engagement" timer and raises an on-screen attention alert if the subject stays disengaged past a threshold.
- Overlays a live HUD: current state, confidence score, engagement group, and session timer.

## 🧠 Research / model development

The `notebooks/` directory contains the full experimental pipeline used to train and evaluate the classical ML models on the extracted landmark features, including:
- Dataset preparation and class mapping from an image-folder structure (Engaged: Focused / Frustrated / Confused, Not Engaged: Looking Away / Drowsy / Bored).
- Training and validation curves for **XGBoost** (log-loss, accuracy).
- **Precision, Recall, and AUC-ROC** curves (one-vs-rest, per class).
- **Precision–Recall curves** with average precision per class.
- Feature importance analysis (native XGBoost/LightGBM importance and permutation importance) for the **Hybrid (SVM + XGBoost)** model.
- Side-by-side comparison of SVM and Hybrid model feature contributions.

The deployed real-time application (`src/realtime_attentiveness_monitor.py`) uses the best-performing landmark-based model (Random Forest) fused with a separately trained EfficientNet CNN for the final production pipeline.

## 📁 Repository structure

```
.
├── src/
│   └── realtime_attentiveness_monitor.py   # Real-time webcam inference app (RF + CNN fusion)
├── notebooks/
│   └── Attentiveness_Detection.ipynb       # Training, evaluation, and visualization pipeline
├── models/                                 # (not tracked) trained model artifacts — see below
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Requirements

- Python 3.9+
- A webcam
- The following trained model artifacts placed in the project root (or update `base_path` in the script):
  - `rf_6class.pkl` — trained Random Forest classifier
  - `le_6class.pkl` — label encoder for the 6 classes
  - `best_effnet.h5` — fine-tuned EfficientNet CNN
  - `class_indices.json` — class index mapping for the CNN
  - `face_landmarker.task` — MediaPipe Face Landmarker model
  - `face_detector.task` — MediaPipe Face Detector model

> Model weight files are not included in this repository (see `.gitignore`). Host them via [Git LFS](https://git-lfs.github.com/), a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github), or a cloud storage link, and reference them here once published.

### Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

```bash
python src/realtime_attentiveness_monitor.py
```

- Press **Q** to quit the monitoring session.
- The on-screen HUD shows the current predicted state, confidence, engagement group, and elapsed session time.
- A red alert bar appears if the subject remains non-engaged beyond ~12 seconds.

## 🔬 Methodology summary

1. **Feature Extraction (Landmark-based):** 478-point face mesh from MediaPipe → eye aspect ratio (EAR), head tilt, brow height, mouth openness, face width/height ratio, gaze centering, and drowsiness heuristic — fed to a Random Forest classifier trained on the 6-class dataset.
2. **Deep Feature Extraction (Image-based):** Cropped and padded face region resized to 224×224, normalized, and passed through a fine-tuned EfficientNet CNN.
3. **Decision-Level Fusion:** Final class probabilities are computed as a weighted sum: `0.35 × RF_probs + 0.65 × CNN_probs`.
4. **Temporal Smoothing:** A 10-frame rolling majority vote stabilizes the displayed label against per-frame jitter.
5. **Engagement Scoring:** A decaying/accumulating timer tracks sustained non-engagement to trigger real-time attention alerts.

## 📊 Evaluation

Model performance (accuracy, precision/recall, AUC-ROC, and feature importance) is documented in the training notebook (`notebooks/Attentiveness_Detection.ipynb`), covering XGBoost, LightGBM, SVM, and Hybrid (SVM + XGBoost) classifiers evaluated on held-out validation splits.

## 📄 Citation

Pending...
If you use this work in your research, please cite this repository (citation details / paper reference to be added upon publication).

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) for face detection and landmark extraction.
- [TensorFlow/Keras](https://www.tensorflow.org/) for the EfficientNet CNN backbone.
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [LightGBM](https://lightgbm.readthedocs.io/) for classical ML models.
