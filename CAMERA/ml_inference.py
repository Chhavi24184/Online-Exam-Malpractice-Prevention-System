import cv2
import joblib
import numpy as np
import pandas as pd

# ===============================
# LOAD FACE CASCADE
# ===============================
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# ===============================
# LOAD TRAINED ML PIPELINE
# ===============================
model = joblib.load("online_exam_malpractice_model.pkl")

# ===============================
# FEATURE ORDER (MUST MATCH TRAINING)
# ===============================
FEATURE_ORDER = [
    "face_count",
    "tab_switch",
    "phone_detected",
    "looking_away_time",
    "exam_duration",
    "risk_score",
    "tab_switch_rate",
    "away_time_ratio",
    "behavior_score",
    "multi_face_flag",
    "malpractice_index"
]

# ===============================
# CAMERA INITIALIZATION
# ===============================
def get_camera():
    for index in [0, 1, 2]: # Try common indices for DroidCam or secondary cameras
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Successfully connected to camera at index {index}")
            return cap
    return None

cap = get_camera()
if cap is None:
    print("Error: Could not open any camera. Please check your connection.")
    exit()

no_face_frames = 0
FACE_MISSING_THRESHOLD = 50
exam_duration_simulated = 100 # Simulated duration for rate calculation

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    face_count = len(faces)

    # Draw face boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Presence tracking
    if face_count == 0:
        no_face_frames += 1
    else:
        no_face_frames = 0

    away_time = no_face_frames
    multiple_face_flag = 1 if face_count > 1 else 0

    # Calculate derived features as per notebook
    tab_switch = 0 # Simulated
    phone_detected = 0 # Simulated
    
    risk_score = (
        (face_count - 1) * 20 +
        tab_switch * 3 +
        phone_detected * 25 +
        away_time * 0.5
    )
    
    tab_switch_rate = tab_switch / (exam_duration_simulated + 1)
    away_time_ratio = away_time / (exam_duration_simulated + 1)
    
    behavior_score = (
        tab_switch * 2 +
        phone_detected * 5 +
        away_time * 0.2
    )
    
    malpractice_index = (
        risk_score * 0.6 +
        behavior_score * 0.4
    )

    # ===============================
    # FEATURE DICTIONARY
    # ===============================
    feature_dict = {
        "face_count": face_count,
        "tab_switch": tab_switch,
        "phone_detected": phone_detected,
        "looking_away_time": away_time,
        "exam_duration": exam_duration_simulated,
        "risk_score": risk_score,
        "tab_switch_rate": tab_switch_rate,
        "away_time_ratio": away_time_ratio,
        "behavior_score": behavior_score,
        "multi_face_flag": multiple_face_flag,
        "malpractice_index": malpractice_index
    }

    # Build feature DataFrame in correct order
    features_df = pd.DataFrame([feature_dict])[FEATURE_ORDER]

    # ===============================
    # ML INFERENCE
    # ===============================
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    confidence = np.max(probabilities)

    if prediction == 0:
        status = "NORMAL"
        color = (0, 255, 0)
    elif prediction == 1:
        status = "MALPRACTICE" # In notebook, default=1 is malpractice
        color = (0, 0, 255)
    else:
        status = "SUSPICIOUS"
        color = (0, 165, 255)

    # ===============================
    # RULE-BASED SAFETY OVERRIDE
    # ===============================
    if face_count > 2 or away_time > FACE_MISSING_THRESHOLD:
        status = "MALPRACTICE (RULE)"
        color = (0, 0, 255)

    # ===============================
    # DISPLAY OUTPUT
    # ===============================
    cv2.putText(frame, f"Faces: {face_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.putText(frame, f"Away Frames: {away_time}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.putText(frame, f"Status: {status}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("Online Exam Malpractice Prevention System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
