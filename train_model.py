import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Set seed for reproducibility
np.random.seed(42)
N = 6000

# Generate synthetic data
face_count = np.random.choice([1, 1, 1, 2, 3], size=N)
tab_switch = np.random.randint(0, 10, size=N)
phone_detected = np.random.choice([0, 1], p=[0.8, 0.2], size=N)
looking_away_time = np.random.randint(0, 120, size=N)
exam_duration = np.random.randint(30, 180, size=N)

# Feature engineering (from notebook)
risk_score = (
    (face_count - 1) * 20 +
    tab_switch * 3 +
    phone_detected * 25 +
    looking_away_time * 0.5
)

prob = 1 / (1 + np.exp(-risk_score / 15))
default = np.random.binomial(1, prob)

data = pd.DataFrame({
    "face_count": face_count,
    "tab_switch": tab_switch,
    "phone_detected": phone_detected,
    "looking_away_time": looking_away_time,
    "exam_duration": exam_duration,
    "risk_score": risk_score,
    "default": default
})

# Add noise and derived features as per notebook
noisy_data = data.copy()
face_count_noise = np.random.randint(-1, 2, size=N)
tab_switch_noise = np.random.randint(-2, 3, size=N)
looking_away_time_noise = np.random.randint(-10, 11, size=N)
exam_duration_noise = np.random.randint(-10, 11, size=N)

noisy_data["face_count"] += face_count_noise
noisy_data["tab_switch"] += tab_switch_noise
noisy_data["looking_away_time"] += looking_away_time_noise
noisy_data["exam_duration"] += exam_duration_noise

noisy_data["face_count"] = noisy_data["face_count"].clip(1, 3)
noisy_data["tab_switch"] = noisy_data["tab_switch"].clip(0, 40)
noisy_data["looking_away_time"] = noisy_data["looking_away_time"].clip(0, 400)
noisy_data["exam_duration"] = noisy_data["exam_duration"].clip(10, 180)

noisy_data["risk_score"] = (
    (noisy_data["face_count"] - 1) * 20 +
    noisy_data["tab_switch"] * 3 +
    noisy_data["phone_detected"] * 25 +
    noisy_data["looking_away_time"] * 0.5
)

prob = 1 / (1 + np.exp(-noisy_data["risk_score"] / 15))
noisy_data["default"] = np.random.binomial(1, prob)

noisy_data["tab_switch_rate"] = noisy_data["tab_switch"] / (noisy_data["exam_duration"] + 1)
noisy_data["away_time_ratio"] = noisy_data["looking_away_time"] / (noisy_data["exam_duration"] + 1)

noisy_data["behavior_score"] = (
    noisy_data["tab_switch"] * 2 +
    noisy_data["phone_detected"] * 5 +
    noisy_data["looking_away_time"] * 0.2
)

noisy_data["multi_face_flag"] = (noisy_data["face_count"] > 1).astype(int)

noisy_data["malpractice_index"] = (
    noisy_data["risk_score"] * 0.6 +
    noisy_data["behavior_score"] * 0.4
)

noisy_data = noisy_data.drop_duplicates()

# Prepare features and labels
X = noisy_data.drop(["default"], axis=1)
y = noisy_data["default"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance data using SMOTE
sm = SMOTE(sampling_strategy=0.6, random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Preprocessing pipeline
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder="passthrough"
)

# Use Random Forest (the best performing model in notebook)
model = RandomForestClassifier(
    n_estimators=100, # Reduced for faster training in this script
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

# Final pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Train model
print("Training model...")
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "online_exam_malpractice_model.pkl")
print("Model saved as online_exam_malpractice_model.pkl")
