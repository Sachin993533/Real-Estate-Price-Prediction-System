# collect_faces.py
# Usage: python collect_faces.py Sachin
import cv2, os, sys
from datetime import datetime

name = sys.argv[1] if len(sys.argv) > 1 else "Person"
save_dir = os.path.join("data", "known", name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0
print("[INFO] Press 'c' to capture face image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(frame, f"Saved: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Collect Faces - Press 'c'", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and len(faces)>0:
        (x,y,w,h) = faces[0]
        face_img = frame[y:y+h, x:x+w]
        fname = os.path.join(save_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(fname, face_img)
        count += 1
        print(f"[SAVED] {fname}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
