import cv2
import string_constants


def get_hand_move(hand_landmarks):
    landmarks = hand_landmarks.landmark
    if all([landmarks[i].y < landmarks[i + 3].y for i in range(9, 20, 4)]):
        return string_constants.ROCK
    elif landmarks[13].y < landmarks[16].y and landmarks[17].y < landmarks[20].y:
        return string_constants.SCISSORS
    else:
        return string_constants.PAPER


def draw_multiline_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=1, font_color=(0, 0, 255), thickness=2):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    x, y = position

    lines = text.split('\n')

    for line in lines:
        cv2.putText(image, line, (x, y), font, font_scale, font_color, thickness)
        y += text_size[1] * 1.5
        y = int(y)


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
