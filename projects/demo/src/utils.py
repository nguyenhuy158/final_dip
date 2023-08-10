import cv2
import numpy as np
import mediapipe as mp
from src import string_constants, var
from pynput.keyboard import Controller, Key

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White color
font_thickness = 2
keyboard = Controller()


def check_hand_horizontal(palm_landmarks):
    try:
        # Calculate the slope of the hand line (y2 - y1) / (x2 - x1)
        slope = (palm_landmarks[0][1] - palm_landmarks[9][1]) / (palm_landmarks[0][0] - palm_landmarks[9][0])

        # Define a threshold for the slope to determine if the hand is horizontal
        slope_threshold = 0.5  # Adjust this value as needed

        return abs(slope) < slope_threshold
    except ZeroDivisionError:
        print("Error div zero")
    return False


def put_text_in_middle(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    font_thickness = 2

    # Get the size of the text
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Calculate the coordinates to center the text
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2

    # Put the text in the middle of the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return image


def format_number_lead_zero(num):
    return str(num) if num >= 10 else '0' + str(num)


def add_text_to_image(image, text):
    # Font và thông số
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .5  # Kích thước font
    font_color = (255, 255, 255)  # Màu chữ (trắng)
    font_thickness = 2  # Độ dày của font

    # Lấy kích thước của chữ
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Tọa độ để viết chữ (bên trái phía dưới, cách phía dưới 10px)
    text_x = 10
    text_y = image.shape[0] - 10  # Độ cao của hình ảnh trừ đi 10px

    # Vẽ chữ lên hình ảnh
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)


def draw_hand_bounding_box(frame, hand_landmarks, player_name):
    image_height, image_width, _ = frame.shape
    landmark_list = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        landmark_list.append((x, y))

    x, y, w, h = cv2.boundingRect(np.array(landmark_list))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text_size = cv2.getTextSize(player_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

    border_color = (0, 255, 0)
    cv2.rectangle(frame, (x - 1, y - text_size[1] - var.space_between),
                  (x + text_size[0], y + text_size[1] - 2 * var.space_between - 1),
                  border_color, var.border_fill)

    cv2.putText(frame, player_name,
                (x, y - var.space_between),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2)


def get_hand_move(hand_landmarks):
    landmarks = hand_landmarks.landmark
    if all([landmarks[i].y < landmarks[i + 3].y for i in range(9, 20, 4)]):
        return string_constants.ROCK
    elif landmarks[13].y < landmarks[16].y and landmarks[17].y < landmarks[20].y:
        return string_constants.SCISSORS
    else:
        return string_constants.PAPER


def draw_multiline_text(image, text, fs=1):
    # print(text)
    lines = text.split('\n')
    # Calculate the total height of all text lines
    total_text_height = len(lines) * cv2.getTextSize(lines[0], font, fs, font_thickness)[0][1]

    # Calculate the starting y-coordinate for centering vertically
    start_y = (image.shape[0] - total_text_height) // 2

    for line in lines:
        text_size = cv2.getTextSize(line, font, fs, font_thickness)[0]

        # Calculate the starting x-coordinate for centering horizontally
        start_x = (image.shape[1] - text_size[0]) // 2

        # Put the text line in the image
        cv2.putText(image, line, (start_x, start_y), font, fs, font_color, font_thickness)

        # Move the starting y-coordinate to the next line
        start_y = int(start_y + text_size[1] * 1.5)

    return image


def check_winner(player1, player2):
    if player1 == player2:
        return string_constants.TIED
    elif player1 == string_constants.PAPER and player2 == string_constants.ROCK:
        return string_constants.PLAYER1_WIN
    elif player1 == string_constants.ROCK and player2 == string_constants.SCISSORS:
        return string_constants.PLAYER1_WIN
    elif player1 == string_constants.SCISSORS and player2 == string_constants.PAPER:
        return string_constants.PLAYER1_WIN
    else:
        return string_constants.PLAYER2_WIN


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def draw_landmarks(frame, hand_landmarks):
    mp_drawing.draw_landmarks(
        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )


def game_running(clock, success, game_text, player1, player2, results):
    if 0 <= clock <= 20:
        success = True
        game_text = string_constants.start
    elif clock < 50:
        game_text = f"{5 - clock // 10}..."
    # elif clock < 40:
    #     game_text = "2..."
    # elif clock < 50:
    #     game_text = "1..."
    elif clock < 60:
        game_text = string_constants.process
    elif clock == 60:
        hand_landmarks = results.multi_hand_landmarks
        if hand_landmarks and len(hand_landmarks) == 2:
            player1 = get_hand_move(hand_landmarks[0])
            player2 = get_hand_move(hand_landmarks[1])
        else:
            success = False
            game_text = string_constants.FAIL
    elif clock < var.MAX_TIME:
        if success:
            game_text = f"P1 {player1}.\nP2 {player2}.\n"
            game_text += check_winner(player1, player2)
            print(game_text)

    return clock, success, game_text, player1, player2


def get_current_option(frame):
    results = hands.process(frame)
    if results.multi_hand_landmarks is None:
        return None

    # print(f"len {len(results.multi_hand_landmarks)}")
    if len(results.multi_hand_landmarks) == 1:
        # print(results.multi_hand_landmarks[0])
        hand_landmarks = results.multi_hand_landmarks[0]
        # draw_landmarks(frame, hand_landmarks)
        draw_hand_bounding_box(frame, hand_landmarks, string_constants.PLAYER1)

        if hand_landmarks.landmark[1].y > hand_landmarks.landmark[4].y and \
                hand_landmarks.landmark[5].y > hand_landmarks.landmark[8].y and \
                hand_landmarks.landmark[9].y > hand_landmarks.landmark[12].y and \
                hand_landmarks.landmark[13].y > hand_landmarks.landmark[16].y and \
                hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y:
            return 5

        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y and \
                hand_landmarks.landmark[9].y < hand_landmarks.landmark[12].y and \
                hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and \
                hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y:
            return 1

        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y and \
                hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y and \
                hand_landmarks.landmark[16].y < hand_landmarks.landmark[13].y and \
                hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y:
            return 3

        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y and \
                hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y and \
                hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and \
                hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y:
            return 2
        else:
            return None

    elif len(results.multi_hand_landmarks) == 2:
        add_text_to_image(frame, string_constants.warn_one_hand)
    return None


def detect_number(hand_landmarks):
    if (hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y) \
            and (hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y) \
            and (hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y):
        return string_constants.ROCK


def put_text_horizontal(image, is_horizontal=False):
    cv2.putText(image, string_constants.waiting_to_exit if is_horizontal else string_constants.tutor_to_exit,
                var.position_horizontal_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                var.scale_half, var.white_color, var.thickness_double)


def is_quite_game_1(is_quit, flag_game_1, option, game_clock_1):
    if is_quit:
        return False, False, None, var.MIN_TIME, var.MIN_TIME
    return is_quit, flag_game_1, option, var.MIN_TIME, game_clock_1


def is_quite_game_2(is_quit, flag_game_2, option, game_clock_2):
    if is_quit:
        return False, False, None, var.MIN_TIME, var.MIN_TIME
    return is_quit, flag_game_2, option, var.MIN_TIME, game_clock_2


def is_quite_game_3(is_quit, flag_game_3, option, game_clock_3):
    if is_quit:
        return False, False, None, var.MIN_TIME, var.MIN_TIME
    return is_quit, flag_game_3, option, var.MIN_TIME, game_clock_3


def is_horizontal(frame, frame_hands):
    if len(frame_hands) == 1:
        hand_landmarks = frame_hands[0]
        palm_landmarks = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in
                          hand_landmarks.landmark]

        return check_hand_horizontal(palm_landmarks)
    return False


def change_video_capture_size(video_capture):
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, var.new_width)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, var.new_height)
    pass
