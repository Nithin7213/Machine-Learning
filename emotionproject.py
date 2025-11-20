import cv2
from deepface import DeepFace
import pyttsx3
import threading

engine = pyttsx3.init()

def speak(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait())).start()

cap = cv2.VideoCapture(0)
last_emotion = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    if not isinstance(results, list):
        results = [results]

    for face in results:
        region = face['region']
        x = int(region['x'])
        y = int(region['y'])
        w = int(region['w'])
        h = int(region['h'])

        emotion = face['dominant_emotion']

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        if emotion != last_emotion:
            speak(f"You look {emotion}")
            last_emotion = emotion

    cv2.imshow("Emotion + Voice", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
