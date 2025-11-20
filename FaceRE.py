import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert color format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for point in landmarks.landmark:
                x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # mp_drawing.draw_landmarks(
            #     frame,
            #     landmarks,
            #     mp_face_mesh.FACEMESH_TESSELATION,  # Facial connections
            #     landmark_drawing_spec=None,  # Default landmark style
            #     # connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            #)

    cv2.imshow('Face Landmarks', frame)
    cv2.waitKey(1)

    if cv2.getWindowProperty('Face Landmarks', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
