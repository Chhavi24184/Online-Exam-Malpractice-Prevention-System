import cv2
import joblib

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

model = joblib.load("online_exam_malpractice_model.pkl")

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_count = len(faces)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    if face_count == 0:
        no_face_frames += 1
    else:
        no_face_frames = 0

    malpractice_flag = 1 if (face_count > 1 or no_face_frames > 50) else 0
    away_time = no_face_frames

    features = [[face_count, away_time, malpractice_flag]]
    prediction = model.predict(features)[0]

    status = "NORMAL" if prediction == 0 else \
             "SUSPICIOUS" if prediction == 1 else "MALPRACTICE"

    cv2.putText(frame, f"STATUS: {status}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Exam Monitoring System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
