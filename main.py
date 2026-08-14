import cv2, joblib, json, time, os, warnings
import numpy as np
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.layers import Dense

# ── 1. ENVIRONMENT SETUP ─────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)


class SafeDense(Dense):
    """Handles Keras 3 compatibility for older models."""

    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)


# ── 2. MODEL LOADING & VALIDATION ────────────────────────────
try:
    print("⏳ Initializing Smart Monitor components...")
    base_path = os.path.dirname(os.path.abspath(__file__))


    def get_path(f):
        p = os.path.join(base_path, f)
        if not os.path.exists(p): raise FileNotFoundError(f"Missing: {f}")
        if os.path.getsize(p) == 0: raise EOFError(f"File is 0KB: {f}")
        return p


    # Load Assets
    rf_model = joblib.load(get_path('rf_6class.pkl'))
    le_6 = joblib.load(get_path('le_6class.pkl'))
    cnn = tf.keras.models.load_model(get_path('best_effnet.h5'),
                                     custom_objects={'Dense': SafeDense}, compile=False)

    with open(get_path('class_indices.json')) as f:
        raw_idx = json.load(f)
        idx2cls = {int(k) if str(k).isdigit() else v: (v if str(k).isdigit() else k)
                   for k, v in raw_idx.items()}

    # MediaPipe Initializers
    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=get_path('face_landmarker.task')),
            running_mode=mp_vision.RunningMode.IMAGE, num_faces=1))

    det_obj = mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=get_path('face_detector.task')),
            running_mode=mp_vision.RunningMode.IMAGE))

    print("✅ All systems online!")

except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}\nEnsure face_detector.task is ~4MB, not 0KB.")
    exit()

# ── 3. CONFIGURATION ─────────────────────────────────────────
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
L_BROW, R_BROW = [336, 296, 334, 293, 300], [107, 66, 105, 63, 70]

CLASS_CFG = {
    'focused': ('FOCUSED', (0, 220, 80), 'ENGAGED'),
    'frustrated': ('FRUSTRATED', (0, 180, 255), 'ENGAGED'),
    'confused': ('CONFUSED', (180, 160, 0), 'ENGAGED'),
    'looking_away': ('LOOKING AWAY', (0, 80, 255), 'NON-ENGAGED'),
    'drowsy': ('DROWSY', (60, 0, 220), 'NON-ENGAGED'),
    'bored': ('BORED', (120, 50, 180), 'NON-ENGAGED'),
}
ENGAGED_CLASSES = {'focused', 'frustrated', 'confused'}


# ── 4. CORE LOGIC ────────────────────────────────────────────
def ear_calc(lm, idx):
    p = [(lm[i].x, lm[i].y) for i in idx]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3])) + 1e-6
    return (A + B) / (2.0 * C)


def predict_6class(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    rf_p = np.array([1 / 6] * 6);
    cnn_p = np.array([1 / 6] * 6);
    bbox = (0, 0, 0, 0)

    # 1. Feature Extraction (RF)
    lm_res = landmarker.detect(mi)
    if lm_res.face_landmarks:
        lm = lm_res.face_landmarks[0]
        l_ear, r_ear = ear_calc(lm, LEFT_EYE), ear_calc(lm, RIGHT_EYE)
        avg_ear = (l_ear + r_ear) / 2
        feats = [lm[1].x, lm[1].y, lm[152].y, avg_ear, abs(lm[LEFT_EYE[0]].y - lm[RIGHT_EYE[0]].y),
                 abs(lm[13].y - lm[14].y), abs(lm[234].x - lm[454].x) / (abs(lm[1].y - lm[152].y) + 1e-6),
                 l_ear, r_ear, abs(lm[234].x - lm[454].x), np.mean([lm[i].y for i in L_BROW]),
                 abs(lm[L_BROW[0]].x - lm[R_BROW[0]].x), 1.0 if avg_ear < 0.22 else 0.0,
                 1.0 - abs(lm[1].x - (lm[234].x + lm[454].x) / 2) / (abs(lm[234].x - lm[454].x) + 1e-6),
                 (lm[61].y + lm[291].y) / 2]
        rf_p = rf_model.predict_proba([feats])[0]

    # 2. Image Processing (CNN)
    det_res = det_obj.detect(mi)
    if det_res.detections:
        bb = det_res.detections[0].bounding_box
        x1, y1 = max(0, int(bb.origin_x - 0.1 * bb.width)), max(0, int(bb.origin_y - 0.1 * bb.height))
        x2, y2 = min(w, int(bb.origin_x + bb.width * 1.1)), min(h, int(bb.origin_y + bb.height * 1.1))
        bbox = (x1, y1, x2, y2)
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            face_img = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (224, 224))
            inp = np.expand_dims(face_img.astype(np.float32) / 255.0, axis=0)
            cnn_p = cnn.predict(inp, verbose=0)[0]

    # 3. Decision Fusion
    cnn_cls = [idx2cls[i] for i in range(len(idx2cls))]
    rf_aligned = np.array([rf_p[list(le_6.classes_).index(c)] if c in le_6.classes_ else 1 / 6 for c in cnn_cls])
    fused = (0.35 * rf_aligned) + (0.65 * cnn_p)  # Weighted Fusion
    idx = np.argmax(fused)
    return cnn_cls[idx], fused[idx], bbox


# ── 5. MAIN EXECUTION LOOP ───────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

history = []
non_eng_secs = 0
start_time = time.time()

print("\n▶ Press 'Q' to quit monitoring.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)  # Mirror for natural feedback

    label, conf, (x1, y1, x2, y2) = predict_6class(frame)

    # Prediction Smoothing
    history.append(label)
    if len(history) > 10: history.pop(0)
    smooth_label = max(set(history), key=history.count)

    cfg = CLASS_CFG.get(smooth_label, (smooth_label.upper(), (200, 200, 200), 'UNKNOWN'))

    # Engagement Scoring
    if smooth_label not in ENGAGED_CLASSES:
        non_eng_secs += 0.12  # Time increment based on loop speed
    else:
        non_eng_secs = max(0, non_eng_secs - 0.4)  # Recovery speed

    # HUD Rendering
    if x2 > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), cfg[1], 2)

    # Bottom UI Bar
    cv2.rectangle(frame, (0, 0), (640, 75), (20, 20, 20), -1)
    cv2.putText(frame, f"{cfg[0]}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cfg[1], 2)
    cv2.putText(frame, f"{cfg[2]} | Conf: {conf * 100:.1f}%", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Session Timer
    elapsed = int(time.time() - start_time)
    cv2.putText(frame, f"Session: {elapsed // 60:02d}:{elapsed % 60:02d}", (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Focus Alert
    if non_eng_secs > 12:
        cv2.rectangle(frame, (0, 75), (640, 115), (0, 0, 180), -1)
        cv2.putText(frame, f"⚠ ATTENTION: Focus on your task! ({int(non_eng_secs)}s)",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Smart Monitor — Pro v2.0", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()