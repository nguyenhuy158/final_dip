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
clock = string_constants.MIN_TIME
game_clock1 = 0
success = True
player1 = player2 = None
game_text = ''
flag_game_1 = False
while videoCapture.isOpened():
    isReadSuccess, frame = videoCapture.read()

    if not isReadSuccess or frame is None:
        print(string_constants.CAM_ERROR)
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    # print(f"option -> {option}")
    # print(f"clock -> {clock}")
    # print(f"flag_game_1 -> {flag_game_1}")

    if option is None and not flag_game_1:
        utils.draw_multiline_text(frame, menu_text, 0.5)
        option = utils.get_current_option(frame)

    if option == 1 and not flag_game_1:
        utils.add_text_to_image(frame, f"{string_constants.time}: {utils.format_number_lead_zero(clock)}")

        if clock < 0:
            flag_game_1 = True
        else:
            option = utils.get_current_option(frame)

        clock = (clock - 1)

    if flag_game_1:
        game_clock1, success, game_text, player1, player2 = games.rock_paper_scissors(frame, game_clock1, success,
                                                                                      game_text,
                                                                                      player1, player2)
    cv2.imshow(string_constants.window_name, frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()

# END PROCESS
