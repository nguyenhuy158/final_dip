# IMPORT LIBRARY
import cv2
import utils
import mediapipe as mp
from string_constants import menu_text

# MAIN PROCESS
videoCapture = cv2.VideoCapture(0)

# Define the menu option text and position
menu_position = (20, 30)
menu_font = cv2.FONT_HERSHEY_SIMPLEX
menu_font_scale = 0.7
menu_color = (255, 255, 255)  # White color

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# hands = mp_hands.Hands(model_complexity=0,
#                        min_tracking_confidence=0.5
#                        , min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

while videoCapture.isOpened():
    isReadSuccess, frame = videoCapture.read()

    if not isReadSuccess or frame is None:
        print("Camera not found.")
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    utils.draw_multiline_text(frame, menu_text, menu_position)

    # Process the frame to detect hands
    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Process hand landmarks to detect gestures
            # Here, you can analyze the hand landmarks to recognize gestures
            # For example, detecting open/closed fingers to recognize gestures
            # ...
            pass

    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()

# END PROCESS
