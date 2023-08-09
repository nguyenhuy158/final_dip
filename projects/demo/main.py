# IMPORT LIBRARY
import cv2

import games
import utils
import mediapipe as mp
import string_constants
from string_constants import menu_text

# MAIN PROCESS
videoCapture = cv2.VideoCapture(0)

# Define the menu option text and position
menu_position = (20, 30)
menu_font = cv2.FONT_HERSHEY_SIMPLEX
menu_font_scale = 0.7
menu_color = (255, 255, 255)  # White color

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

option = 1
clock = 0
success = True
player1 = player2 = None
game_text = ''

while videoCapture.isOpened():
    isReadSuccess, frame = videoCapture.read()

    if not isReadSuccess or frame is None:
        print(string_constants.CAM_ERROR)
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    if option is None:
        utils.draw_multiline_text(frame, menu_text, 0.5)

    if option == 1:
        clock, success, game_text, player1, player2 = games.rock_paper_scissors(frame, clock, success, game_text,
                                                                                player1, player2)

    cv2.imshow(string_constants.window_name, frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()

# END PROCESS
