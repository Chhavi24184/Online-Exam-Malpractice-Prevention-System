import cv2

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

    # Face missing logic
    if face_count == 0:
        no_face_frames += 1
    else:
        no_face_frames = 0

    malpractice_flag = 1 if (face_count > 1 or no_face_frames > 50) else 0

    cv2.putText(frame, f"Faces: {face_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    if malpractice_flag:
        cv2.putText(frame, "SUSPICIOUS ACTIVITY",
                    (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,0,255), 2)

    cv2.imshow("Exam Monitoring", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
