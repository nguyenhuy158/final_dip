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

clock = 0
success = True
player1 = player2 = None
gameText = ''

while videoCapture.isOpened():
    isReadSuccess, frame = videoCapture.read()

    if not isReadSuccess or frame is None:
        print("Camera not found.")
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    # utils.draw_multiline_text(frame, menu_text, menu_position)

    # Process the frame to detect hands
    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    if 0 <= clock <= 20:
        success = True
        gameText = "Bat dau"
    elif clock < 30:
        gameText = "3..."
    elif clock < 40:
        gameText = "2..."
    elif clock < 50:
        gameText = "1..."
    elif clock < 60:
        gameText = "Xu ly!"
    elif clock == 60:
        hand_landmarks = results.multi_hand_landmarks
        if hand_landmarks and len(hand_landmarks) == 2:
            player1 = utils.get_hand_move(hand_landmarks[0])
            player2 = utils.get_hand_move(hand_landmarks[1])
        else:
            success = False
    elif clock < 100:
        if success:
            gameText = f"P1 {player1}. P2 {player2}."
            gameText += utils.check_winner(player1, player2)
            print(gameText)

    utils.draw_multiline_text(frame, gameText, (50, 50))
    utils.draw_multiline_text(frame, f"Thoi Gian: {clock}", (50, 80))

    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    clock = (clock + 1) % 100

videoCapture.release()
cv2.destroyAllWindows()

# END PROCESS
