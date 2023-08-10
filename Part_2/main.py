# IMPORT LIBRARY
import collections
import cv2
import numpy as np
import games
from src import utils, var, string_constants, quick_draw_utils
import config
import mediapipe as mp
from src.string_constants import menu_text

# END IMPORT LIBRARY


# BEGIN GAMES
videoCapture = cv2.VideoCapture(0)
utils.change_video_capture_size(videoCapture)

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
clock = var.MIN_TIME

game_play_clock1 = 0
game_clock1 = var.MIN_TIME
game_clock2 = var.MIN_TIME
game_clock3 = var.MIN_TIME
success = True
player1 = player2 = None
game_text = ''
flag_game_1 = False
flag_game_2 = False
flag_game_3 = False
is_quit = False

# quickdraw
cap = cv2.VideoCapture(1)
points = collections.deque(maxlen=512)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
is_drawing = False
is_shown = False
class_images = quick_draw_utils.get_images("images", config.CLASSES)


# LIST FUNCTION
def logging():
    print(f"option -> {option}")
    print(f"clock -> {clock}")
    print(f"flag_game_1 -> {flag_game_1}")
    print(f"flag_game_2 -> {flag_game_2}")
    print(f"flag_game_3 -> {flag_game_3}")
    print(f"game_clock1 -> {game_clock1}")
    print(f"game_clock2 -> {game_clock2}")
    print(f"is_quit -> {is_quit}")


# END LIST FUNCTION


# MAIN GAMES
while videoCapture.isOpened():
    isReadSuccess, frame = videoCapture.read()

    if not isReadSuccess or frame is None:
        print(string_constants.CAM_ERROR)
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    # frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    backup_frame = frame.copy()
    text_frame = np.zeros_like(frame)

    # logging()

    if option is None or option == 5 and not flag_game_1 and not flag_game_2 and not flag_game_3:
        utils.draw_multiline_text(text_frame, menu_text)
        option = utils.get_current_option_and_warming(frame, text_frame)

    # GAME ONE
    if option == 1 and not flag_game_1:
        utils.add_normal_text_to_right_bottom(text_frame,
                                              f"{string_constants.time}: {utils.format_number_lead_zero(clock)}")
        if clock < 0:
            flag_game_1 = True
        else:
            new_option = utils.get_current_option_and_warming(frame, text_frame)
            clock = var.MIN_TIME if new_option == 2 or new_option == 5 or new_option == 3 else clock
            option = new_option
        clock = (clock - 1)
    # GAME TWO
    if option == 2 and not flag_game_2:
        utils.add_normal_text_to_right_bottom(text_frame,
                                              f"{string_constants.time}: {utils.format_number_lead_zero(clock)}")
        if clock < 0:
            flag_game_2 = True
        else:
            new_option = utils.get_current_option_and_warming(frame, text_frame)
            clock = var.MIN_TIME if new_option == 1 or new_option == 5 or new_option == 3 else clock
            option = new_option
        clock = (clock - 1)
    # GAME THREE
    if option == 3 and not flag_game_3:
        utils.add_normal_text_to_right_bottom(text_frame,
                                              f"{string_constants.time}: {utils.format_number_lead_zero(clock)}")
        if clock < 0:
            flag_game_3 = True
        else:
            new_option = utils.get_current_option_and_warming(frame, text_frame)
            clock = var.MIN_TIME if new_option == 1 or new_option == 5 or new_option == 2 else clock
            option = new_option
        clock = (clock - 1)
    # PROCESSING GAME
    if flag_game_1:
        game_play_clock1, success, game_text, player1, player2, game_clock1, is_quit \
            = games.rock_paper_scissors(frame,
                                        text_frame,
                                        game_play_clock1,
                                        success,
                                        game_text,
                                        player1,
                                        player2,
                                        game_clock1,
                                        is_quit)
        is_quit, flag_game_1, option, clock, game_clock1 \
            = utils.is_quite_game(is_quit,
                                  flag_game_1,
                                  option,
                                  game_clock1)
    if flag_game_2:
        game_clock2, is_quit \
            = games.dino(frame,
                         text_frame,
                         game_clock2,
                         is_quit)
        is_quit, flag_game_2, option, clock, game_clock2 \
            = utils.is_quite_game(is_quit,
                                  flag_game_2,
                                  option,
                                  game_clock2)
    if flag_game_3:
        points, canvas, is_drawing, is_shown, game_clock3, is_quit \
            = games.quick_draw(frame,
                               text_frame,
                               points,
                               canvas,
                               is_drawing,
                               is_shown,
                               game_clock3,
                               is_quit)
        is_quit, flag_game_3, option, clock, game_clock3 \
            = utils.is_quite_game(is_quit,
                                  flag_game_3,
                                  option,
                                  game_clock3)

    # SHOWING GAME
    result = cv2.addWeighted(text_frame, var.alpha_text, frame, var.beta_frame, 0)
    cv2.imshow(string_constants.window_name, result)

    # HANDLE EXIT GAMES
    if cv2.waitKey(25) & 0xFF == ord("q") or cv2.waitKey(5) & 0xFF == 27:
        break

# RELEASE SOURCE
videoCapture.release()
cv2.destroyAllWindows()
# END GAMES
