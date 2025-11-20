import cv2
import mediapipe as mp
import smtplib
import os
from email.message import EmailMessage
from datetime import datetime

# ============ EMAIL CONFIG ============
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "nithikuttan29@gmail.com"          # <- change
EMAIL_PASSWORD = "oase tvjj tcku poma"      # <- change (Gmail app password)
EMAIL_RECEIVER = "nithinabraham7213@gmail.com"  # <- change

def send_email_with_image(image_path):
    msg = EmailMessage()
    msg["Subject"] = "Pose Detected - Alert"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content("A human pose was detected. Image is attached.")

    with open(image_path, "rb") as f:
        img_data = f.read()
        img_name = os.path.basename(image_path)
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=img_name)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        print("Email sent!")

# ============ MEDIAPIPE / OPENCV SETUP ============
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

email_sent = False  # send only once per run

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # draw pose
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # capture & send only once
        if not email_sent:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"pose_detected_{timestamp}.jpg"
            cv2.imwrite(image_filename, frame)
            print(f"Saved image: {image_filename}")
            send_email_with_image(image_filename)
            email_sent = True

    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()