# Online Exam Malpractice Prevention System

An AI-powered real-time monitoring system designed to prevent and detect malpractice during online examinations using Computer Vision and Machine Learning.

## 📝 Project Summary
This project provides a robust solution for maintaining the integrity of online exams. By leveraging OpenCV for face detection and a trained Random Forest model for behavior analysis, the system can automatically flag suspicious activities such as:
- **Multiple Faces**: Detecting more than one person in the camera frame.
- **Looking Away**: Tracking how long a student's face is missing from the frame.
- **Suspicious Behavior**: Using a machine learning model to analyze various features and assign a risk score.

## 🚀 Features
- **Real-time Monitoring**: Continuous video analysis during the exam.
- **Face Tracking**: Detects and boxes faces in real-time.
- **ML-Based Prediction**: Classifies behavior as `NORMAL`, `SUSPICIOUS`, or `MALPRACTICE`.
- **Automatic Fixes**: Includes a re-training script to ensure environment compatibility.
- **Robust Camera Support**: Automatically detects working cameras (including DroidCam).

## 🛠️ Tech Stack
- **Python**: Primary programming language.
- **OpenCV**: For computer vision and face detection.
- **Scikit-learn**: For machine learning model implementation.
- **Pandas & Numpy**: For data manipulation and feature engineering.
- **Joblib**: For model persistence.

## 📁 Project Structure
- `CAMERA/`: Contains all real-time monitoring scripts.
  - `ml_inference.py`: The main monitoring engine (enhanced for robust camera detection).
  - `exam_monitor.py`: Real-time status tracker.
  - `face_detection.py`: Basic face detection logic.
- `Online_Exam_Malpractice_Prevention.ipynb`: The research and training notebook.
- `train_model.py`: A script to re-generate the model locally to fix version mismatches.
- `online_exam_malpractice_model.pkl`: The trained machine learning model.

## ⚙️ Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install opencv-python joblib scikit-learn pandas numpy imbalanced-learn
   ```

2. **Re-train the Model (Optional but recommended for compatibility)**:
   ```bash
   python train_model.py
   ```

3. **Run the Monitoring System**:
   ```bash
   python CAMERA/ml_inference.py
   ```

## 📜 License
This project is for educational purposes and is intended to help improve the security of online examination platforms.
