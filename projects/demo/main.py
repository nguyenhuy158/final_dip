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
game_clock2 = string_constants.MIN_TIME
success = True
player1 = player2 = None
game_text = ''
flag_game_1 = False
flag_game_2 = False
is_quit = False

while videoCapture.isOpened():
    isReadSuccess, frame = videoCapture.read()

    if not isReadSuccess or frame is None:
        print(string_constants.CAM_ERROR)
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    print(f"option -> {option}")
    print(f"clock -> {clock}")
    print(f"flag_game_1 -> {flag_game_1}")
    print(f"flag_game_2 -> {flag_game_2}")
    print(f"game_clock1 -> {game_clock1}")
    print(f"game_clock2 -> {game_clock2}")
    print(f"is_quit -> {is_quit}")

    if option is None or option == 5 and not flag_game_1 and not flag_game_2:
        utils.draw_multiline_text(frame, menu_text, 0.5)
        option = utils.get_current_option(frame)

    if option == 1 and not flag_game_1:
        utils.add_text_to_image(frame, f"{string_constants.time}: {utils.format_number_lead_zero(clock)}")

        if clock < 0:
            flag_game_1 = True
        else:
            new_option = utils.get_current_option(frame)
            clock = string_constants.MIN_TIME if new_option == 2 or new_option == 5 else clock
            option = new_option

        clock = (clock - 1)

    if option == 2 and not flag_game_2:
        utils.add_text_to_image(frame, f"{string_constants.time}: {utils.format_number_lead_zero(clock)}")

        if clock < 0:
            flag_game_2 = True
        else:
            new_option = utils.get_current_option(frame)
            clock = string_constants.MIN_TIME if new_option == 1 or new_option == 5 else clock
            option = new_option

        clock = (clock - 1)

    if flag_game_1:
        game_clock1, success, game_text, player1, player2, is_quit = games.rock_paper_scissors(frame, game_clock1,
                                                                                               success,
                                                                                               game_text,
                                                                                               player1, player2,
                                                                                               is_quit)
        is_quit, flag_game_2, option, clock, game_clock2 = utils.is_quite_game_2(is_quit, flag_game_2, option,
                                                                                 game_clock2)

    if flag_game_2:
        game_clock2, is_quit = games.dino(frame, game_clock2, is_quit)
        is_quit, flag_game_2, option, clock, game_clock2 = utils.is_quite_game_2(is_quit, flag_game_2, option,
                                                                                 game_clock2)

    cv2.imshow(string_constants.window_name, frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()

# END PROCESS
