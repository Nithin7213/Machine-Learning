# ai_gym_trainer.py
"""
AI Gym Trainer — Single-file (Futuristic HUD + 6 exercises)

Dependencies:
    pip install mediapipe opencv-python numpy pyttsx3

Run:
    python ai_gym_trainer.py
"""
import time
import threading
from collections import deque, defaultdict

import cv2
import mediapipe as mp
import numpy as np

# Try import pyttsx3, but allow running if not available (will only disable voice)
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# ---------------------- Settings ----------------------
VISIBILITY_THRESHOLD = 0.55        # landmark visibility threshold to consider "ready"
ANGLE_SMOOTHING_WINDOW = 5        # sliding window size for smoothing angles
VOICE_RATE = 160                  # speech rate
VOICE_ENABLED = True              # default voice state

# ---------------------- Utilities ----------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Voice class that runs in background thread
class VoiceFeedback:
    def __init__(self, rate=VOICE_RATE):
        self.enabled = True
        self.engine = None
        if pyttsx3:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", rate)
            except Exception:
                self.engine = None
        self.queue = deque()
        self._running = True
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self):
        while self._running:
            if self.queue and self.engine and self.enabled:
                text = self.queue.popleft()
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception:
                    # swallow pyttsx3 issues
                    pass
            else:
                time.sleep(0.04)

    def say(self, text):
        if not text:
            return
        # limit queue length
        if len(self.queue) < 6:
            self.queue.append(text)

    def stop(self):
        self._running = False
        try:
            if self.engine:
                self.engine.stop()
        except Exception:
            pass

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

# convert mediapipe landmark to scaled x,y,z tuple
def landmark_point(lm, w, h):
    # scale z by width so units roughly comparable to x/y in pixels
    return (lm.x * w, lm.y * h, lm.z * w)

# 3D angle at point b formed by points a-b-c
def angle_3d(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.dot(ba, bc) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

# smoothing buffers for named angles
_angle_buffers = defaultdict(lambda: deque(maxlen=ANGLE_SMOOTHING_WINDOW))

def smooth(name, value):
    buf = _angle_buffers[name]
    buf.append(value)
    return float(sum(buf) / len(buf))

# body readiness check: required is a list of PoseLandmark enums
def body_ready(landmarks, required):
    for lm_enum in required:
        lm = landmarks[lm_enum.value]
        if lm.visibility < VISIBILITY_THRESHOLD:
            return False
    return True

# ---------------------- Exercise base ----------------------
class ExerciseBase:
    name = "Base"

    def __init__(self, voice: VoiceFeedback):
        self.count = 0
        self.stage = None
        self.voice = voice
        self._last_voice = 0.0
        self.voice_interval = 0.9  # seconds minimum between voice hints

    def reset(self):
        self.count = 0
        self.stage = None

    def should_speak(self):
        now = time.time()
        if now - self._last_voice > self.voice_interval:
            self._last_voice = now
            return True
        return False

    def update(self, landmarks, w, h):
        """
        Compute exercise-specific info.
        Must return a dict:
          {
            'exercise': self.name,
            'ready': bool,
            'count': int,
            'stage': str,
            'advice': str (optional),
            ... other angle fields
          }
        """
        raise NotImplementedError

# ---------------------- Exercises ----------------------
# We'll refer to PoseLandmark enum values by .value when indexing the list of landmarks.

class Squat(ExerciseBase):
    name = "Squat"
    required = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE]

    def update(self, landmarks, w, h):
        if not body_ready(landmarks, Squat.required):
            return {'exercise': self.name, 'ready': False, 'count': self.count, 'stage': self.stage, 'advice': 'Full body not visible'}

        # left leg
        l_hip = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], w, h)
        l_knee = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], w, h)
        l_ankle = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value], w, h)
        # right leg
        r_hip = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], w, h)
        r_knee = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], w, h)
        r_ankle = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value], w, h)

        left_knee_angle = angle_3d(l_hip, l_knee, l_ankle)
        right_knee_angle = angle_3d(r_hip, r_knee, r_ankle)
        knee_angle = smooth('squat_knee', (left_knee_angle + right_knee_angle) / 2.0)

        # back/torso tilt: shoulder - hip - knee
        l_sh = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        r_sh = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        sh = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0, (l_sh[2] + r_sh[2]) / 2.0)
        hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0, (l_hip[2] + r_hip[2]) / 2.0)
        knee = ((l_knee[0] + r_knee[0]) / 2.0, (l_knee[1] + r_knee[1]) / 2.0, (l_knee[2] + r_knee[2]) / 2.0)
        back_angle = smooth('squat_back', angle_3d(sh, hip, knee))

        # state machine thresholds (heuristic, adjustable)
        if knee_angle > 160:
            new_stage = 'up'
        elif knee_angle < 120:
            new_stage = 'down'
        else:
            new_stage = self.stage or 'up'

        advice = None
        # count rep only when down -> up and back posture is good
        if self.stage == 'down' and new_stage == 'up':
            if back_angle > 140:
                self.count += 1
                advice = 'Good squat'
                if self.should_speak() and self.voice:
                    self.voice.say('Good squat')
            else:
                advice = 'Keep your back straight to count rep'
                if self.should_speak() and self.voice:
                    self.voice.say('Keep your back straight')

        # live hints
        if back_angle < 140:
            advice = 'Keep your back straight'
        elif knee_angle > 140:
            advice = 'Squat lower'
        elif knee_angle <= 120:
            advice = 'Hold low and come up'

        self.stage = new_stage
        return {'exercise': self.name, 'ready': True, 'count': self.count, 'stage': self.stage,
                'advice': advice, 'knee_angle': knee_angle, 'back_angle': back_angle}

class Pushup(ExerciseBase):
    name = "Pushup"
    required = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP]

    def update(self, landmarks, w, h):
        if not body_ready(landmarks, Pushup.required):
            return {'exercise': self.name, 'ready': False, 'count': self.count, 'stage': self.stage, 'advice': 'Full body not visible'}

        l_sh = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        l_el = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], w, h)
        l_wr = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], w, h)
        r_sh = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        r_el = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], w, h)
        r_wr = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], w, h)

        left_elbow_angle = angle_3d(l_sh, l_el, l_wr)
        right_elbow_angle = angle_3d(r_sh, r_el, r_wr)
        elbow_angle = smooth('pushup_elbow', (left_elbow_angle + right_elbow_angle) / 2.0)

        l_hip = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], w, h)
        r_hip = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], w, h)
        l_ank = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value], w, h)
        r_ank = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value], w, h)
        sh = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0, (l_sh[2] + r_sh[2]) / 2.0)
        hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0, (l_hip[2] + r_hip[2]) / 2.0)
        ank = ((l_ank[0] + r_ank[0]) / 2.0, (l_ank[1] + r_ank[1]) / 2.0, (l_ank[2] + r_ank[2]) / 2.0)
        body_angle = smooth('pushup_body', angle_3d(sh, hip, ank))

        if elbow_angle < 100:
            new_stage = 'down'
        elif elbow_angle > 160:
            new_stage = 'up'
        else:
            new_stage = self.stage or 'up'

        advice = None
        if self.stage == 'down' and new_stage == 'up':
            if body_angle > 155:
                self.count += 1
                if self.should_speak() and self.voice:
                    self.voice.say('Good pushup')
            else:
                advice = 'Keep your body straight'
                if self.should_speak() and self.voice:
                    self.voice.say('Keep your body straight')

        if body_angle < 150:
            advice = 'Avoid sagging'
        elif elbow_angle < 120:
            advice = 'Lower more'

        self.stage = new_stage
        return {'exercise': self.name, 'ready': True, 'count': self.count, 'stage': self.stage,
                'advice': advice, 'elbow_angle': elbow_angle, 'body_angle': body_angle}

class BicepCurl(ExerciseBase):
    name = "Bicep Curl"
    required = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST]

    def update(self, landmarks, w, h):
        if not body_ready(landmarks, BicepCurl.required):
            return {'exercise': self.name, 'ready': False, 'count': self.count, 'stage': self.stage, 'advice': 'Upper body not visible'}

        l_sh = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        l_el = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], w, h)
        l_wr = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], w, h)
        r_sh = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        r_el = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], w, h)
        r_wr = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], w, h)

        left_elbow = angle_3d(l_sh, l_el, l_wr)
        right_elbow = angle_3d(r_sh, r_el, r_wr)
        elbow_angle = smooth('bicep_elbow', (left_elbow + right_elbow) / 2.0)

        # typical curling: up (small angle) <-> down (large angle)
        if elbow_angle > 160:
            new_stage = 'down'   # arm extended
        elif elbow_angle < 50:
            new_stage = 'up'     # fully curled
        else:
            new_stage = self.stage or 'down'

        advice = None
        if self.stage == 'up' and new_stage == 'down':
            # completed one down->up? we choose to count up->down or down->up; here count when up->down
            # to ensure consistent counting we count when returning to down (arm extended)
            self.count += 1
            if self.should_speak() and self.voice:
                self.voice.say('One curl')

        if elbow_angle > 160:
            advice = 'Extend your arm fully'
        elif elbow_angle < 60:
            advice = 'Squeeze biceps at top'

        self.stage = new_stage
        return {'exercise': self.name, 'ready': True, 'count': self.count, 'stage': self.stage,
                'advice': advice, 'elbow_angle': elbow_angle}

class ShoulderPress(ExerciseBase):
    name = "Shoulder Press"
    required = BicepCurl.required

    def update(self, landmarks, w, h):
        if not body_ready(landmarks, ShoulderPress.required):
            return {'exercise': self.name, 'ready': False, 'count': self.count, 'stage': self.stage, 'advice': 'Upper body not visible'}

        # estimate shoulder angle by elbow - shoulder - wrist
        l_sh = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        l_el = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], w, h)
        l_wr = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], w, h)
        r_sh = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        r_el = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], w, h)
        r_wr = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], w, h)

        left_angle = angle_3d(l_el, l_sh, l_wr)
        right_angle = angle_3d(r_el, r_sh, r_wr)
        shoulder_angle = smooth('should_press', (left_angle + right_angle) / 2.0)

        # small angle when arms up, large when down depending on geometry; tune thresholds
        if shoulder_angle < 40:
            new_stage = 'up'
        elif shoulder_angle > 85:
            new_stage = 'down'
        else:
            new_stage = self.stage or 'down'

        advice = None
        if self.stage == 'down' and new_stage == 'up':
            self.count += 1
            if self.should_speak() and self.voice:
                self.voice.say('One shoulder press')

        if shoulder_angle < 45:
            advice = 'Arms overhead'
        elif shoulder_angle > 85:
            advice = 'Lower to shoulder level'

        self.stage = new_stage
        return {'exercise': self.name, 'ready': True, 'count': self.count, 'stage': self.stage,
                'advice': advice, 'shoulder_angle': shoulder_angle}

class LateralRaise(ExerciseBase):
    name = "Lateral Raise"
    required = BicepCurl.required

    def update(self, landmarks, w, h):
        if not body_ready(landmarks, LateralRaise.required):
            return {'exercise': self.name, 'ready': False, 'count': self.count, 'stage': self.stage, 'advice': 'Upper body not visible'}

        l_sh = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        l_el = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], w, h)
        l_wr = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], w, h)
        r_sh = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        r_el = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], w, h)
        r_wr = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], w, h)

        left_angle = angle_3d(l_sh, l_el, l_wr)
        right_angle = angle_3d(r_sh, r_el, r_wr)
        raise_angle = smooth('lat_raise', (left_angle + right_angle) / 2.0)

        if raise_angle < 50:
            new_stage = 'up'
        elif raise_angle > 95:
            new_stage = 'down'
        else:
            new_stage = self.stage or 'down'

        advice = None
        if self.stage == 'down' and new_stage == 'up':
            self.count += 1
            if self.should_speak() and self.voice:
                self.voice.say('One lateral raise')

        if raise_angle < 50:
            advice = 'Arms raised to shoulder level'
        elif raise_angle > 95:
            advice = 'Lower slowly'

        self.stage = new_stage
        return {'exercise': self.name, 'ready': True, 'count': self.count, 'stage': self.stage,
                'advice': advice, 'raise_angle': raise_angle}

class SitUp(ExerciseBase):
    name = "Sit-up"
    required = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE]

    def update(self, landmarks, w, h):
        if not body_ready(landmarks, SitUp.required):
            return {'exercise': self.name, 'ready': False, 'count': self.count, 'stage': self.stage, 'advice': 'Torso not visible'}

        l_sh = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        r_sh = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        sh = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0, (l_sh[2] + r_sh[2]) / 2.0)
        l_hp = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], w, h)
        r_hp = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], w, h)
        hip = ((l_hp[0] + r_hp[0]) / 2.0, (l_hp[1] + r_hp[1]) / 2.0, (l_hp[2] + r_hp[2]) / 2.0)
        l_kn = landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], w, h)
        r_kn = landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], w, h)
        knee = ((l_kn[0] + r_kn[0]) / 2.0, (l_kn[1] + r_kn[1]) / 2.0, (l_kn[2] + r_kn[2]) / 2.0)

        torso_angle = smooth('sit_torso', angle_3d(sh, hip, knee))

        if torso_angle < 100:
            new_stage = 'up'
        elif torso_angle > 140:
            new_stage = 'down'
        else:
            new_stage = self.stage or 'down'

        advice = None
        if self.stage == 'down' and new_stage == 'up':
            self.count += 1
            if self.should_speak() and self.voice:
                self.voice.say('One sit up')

        if torso_angle > 140:
            advice = 'Lie flat, start the sit-up'
        elif torso_angle < 110:
            advice = 'Good sit-up, control descent'

        self.stage = new_stage
        return {'exercise': self.name, 'ready': True, 'count': self.count, 'stage': self.stage,
                'advice': advice, 'torso_angle': torso_angle}

# ---------------------- UI / HUD Drawing ----------------------
EXERCISES = [Squat, Pushup, BicepCurl, ShoulderPress, LateralRaise, SitUp]

def draw_status(frame, info):
    h, w = frame.shape[:2]

    # semi-transparent HUD header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (10, 10, 30), -1)
    frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)

    # neon divider
    cv2.line(frame, (0, 120), (w, 120), (0, 230, 230), 2)

    # glow text helper
    def glow_text(img, text, pos, color=(0, 255, 255), scale=0.9):
        x, y = pos
        # draw subtle shadow layers for glow effect
        for thickness in [6, 4, 2]:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (10, 10, 20), thickness, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 2, cv2.LINE_AA)

    # Main HUD texts
    glow_text(frame, f"EXERCISE: {info.get('exercise')}", (20, 45), (0, 230, 230), 0.95)
    glow_text(frame, f"REPS ({info.get('exercise')}): {info.get('count')}", (20, 86), (0, 180, 255), 0.9)

    # READY indicator
    ready = info.get('ready', False)
    indicator_color = (0, 220, 120) if ready else (0, 60, 180)
    # dark inner circle
    cv2.circle(frame, (w - 90, 60), 30, (20, 20, 30), -1)
    # neon ring
    cv2.circle(frame, (w - 90, 60), 26, indicator_color, 3)
    glow_text(frame, 'READY' if ready else 'NOT READY', (w - 230, 110), indicator_color, 0.7)

    # advice banner
    advice = info.get('advice')
    if advice:
        glow_text(frame, advice.upper(), (200, 45), (255, 210, 50), 0.8)

    # details
    y = 140
    for key in ['knee_angle', 'back_angle', 'elbow_angle', 'body_angle', 'shoulder_angle', 'raise_angle', 'torso_angle']:
        if key in info:
            cv2.putText(frame, f"{key}: {int(info[key])}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22

    # bottom HUD
    cv2.rectangle(frame, (8, h - 62), (w - 8, h - 8), (10, 10, 20), 1)
    cv2.putText(frame, "E=Next   R=Reset   P=ToggleVoice   Q=Quit", (20, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 230), 2, cv2.LINE_AA)

    return frame

# ---------------------- Main App Loop ----------------------
def main():
    global VOICE_ENABLED
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    voice = VoiceFeedback(rate=VOICE_RATE)

    idx = 0
    exercise_obj = EXERCISES[idx](voice)

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                info = exercise_obj.update(lm, w, h)

                # draw skeleton with subtle neon colors
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 190, 200), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(60, 180, 200), thickness=2, circle_radius=2))

                frame = draw_status(frame, info)

                # optionally speak important advice (rate-limited by exercise.should_speak)
                if info.get('ready') and info.get('advice') and exercise_obj.should_speak() and voice.enabled:
                    voice.say(info.get('advice'))

            else:
                cv2.putText(frame, 'No person detected - position yourself in frame', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("AI Gym Trainer — Futuristic", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                idx = (idx + 1) % len(EXERCISES)
                exercise_obj = EXERCISES[idx](voice)
                voice.say(f"Switched to {exercise_obj.name}")
            elif key == ord('r'):
                exercise_obj.reset()
                voice.say("Counters reset")
            elif key == ord('p'):
                VOICE_ENABLED = not VOICE_ENABLED
                voice.enabled = VOICE_ENABLED
                voice.say("Voice enabled" if VOICE_ENABLED else "Voice disabled")

    except KeyboardInterrupt:
        pass
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()
        voice.stop()

if __name__ == "__main__":
    main()
