import cv2
import mediapipe as mp
import smtplib
import os
from email.message import EmailMessage
from datetime import datetime

# ============ EMAIL CONFIG ============
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "midhunem670@gmail.com"         
EMAIL_PASSWORD = "zflu bkeh eqav qvvf"      
EMAIL_RECEIVER = "midhunem470@gmail.com" 

def send_email_with_image(image_path):
    msg = EmailMessage()
    msg["Subject"] = "Pose Detected - Alert"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content("A human pose was detected. Image is attached.")

    html_content = f"""\
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background-color: #ffffff;
                    border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
          <h2 style="color: #333333; margin-top: 0;">🤖 Pose Detection Alert</h2>
          <p>Hi,</p>
          <p>
            Your <strong>Python pose-detection system</strong> just detected a human pose.
          </p>
          <p>
            <strong>Image file:</strong> {os.path.basename(image_path)}
          </p>
          <p>
            The captured frame has been attached to this email as an image.
          </p>
          <hr style="border: none; border-top: 1px solid #dddddd; margin: 20px 0;">
          <p style="font-size: 12px; color: #777777; text-align: center;">
            This is an automated message sent from your pose detection script.<br>
            You can safely ignore this email if it was for testing.
          </p>
        </div>
      </body>
    </html>
    """
    msg.add_alternative(html_content, subtype="html")

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