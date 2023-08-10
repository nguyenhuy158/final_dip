import collections
import config
import cv2
import numpy as np
import torch
from src import utils, var, string_constants, quick_draw_utils


def rock_paper_scissors(frame, text_frame, clock, success, game_text, player1, player2, game_clock1, is_quit):
    results = utils.hands.process(frame)
    # step1: DRAW BOX AND PLAYER NAME
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            utils.draw_landmarks(frame, hand_landmarks)
            player_name = f"{string_constants.PLAYER}{idx + 1}"
            utils.draw_hand_bounding_box(frame, hand_landmarks, player_name)
    # step2: START GAME
    clock, success, game_text, player1, player2 = utils.game_running(clock, success,
                                                                     game_text, player1, player2, results)
    # step3
    utils.draw_multiline_text(text_frame, game_text)
    # step4
    utils.add_warning_text_to_bottom_left(text_frame,
                                          f"{string_constants.game_time}: {utils.format_number_lead_zero(clock)}")
    # step5
    clock = (clock + 1) % var.MAX_TIME
    # step6.1
    is_horizontal = utils.is_horizontal(frame, results.multi_hand_landmarks)
    # step6.2
    utils.put_text_horizontal(text_frame, is_horizontal)
    if is_horizontal and game_clock1 <= 0:
        is_quit = True
    elif is_horizontal:
        game_clock1 -= 1
    else:
        game_clock1 = var.MIN_TIME
    # step6.3
    utils.add_normal_text_to_right_bottom(text_frame, f"{string_constants.time}: {game_clock1}")
    return clock, success, game_text, player1, player2, game_clock1, is_quit


def dino(frame, text_frame, game_clock2, is_quit):
    results = utils.hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # step1
            utils.draw_landmarks(frame, hand_landmarks)
            # step2
            utils.draw_hand_bounding_box(frame, hand_landmarks, string_constants.PLAYER1)
            # step3
            if utils.detect_number(hand_landmarks) == string_constants.ROCK:
                utils.keyboard.press(utils.Key.space)
                cv2.putText(text_frame, "Press", (10, frame.shape[0] - 100), var.font, var.scale_half, var.white_color,
                            var.thickness_double)
            else:
                utils.keyboard.release(utils.Key.space)
            # step4
            is_horizontal = utils.is_horizontal(frame, results.multi_hand_landmarks)
            # step5
            utils.put_text_horizontal(text_frame, is_horizontal)
            if is_horizontal and game_clock2 <= 0:
                is_quit = True
            elif is_horizontal:
                game_clock2 -= 1
            else:
                game_clock2 = var.MIN_TIME
            # step6
            utils.add_normal_text_to_right_bottom(text_frame,
                                                  f"{string_constants.time}: {game_clock2}")

    return game_clock2, is_quit


def quick_draw(frame, text_frame, points, canvas, is_drawing, is_shown, game_clock3, is_quit):
    results = utils.hands.process(frame)

    # Draw the hand annotations on the image.
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print(f"Game quick draw start")
            utils.draw_landmarks(frame, hand_landmarks)

            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y \
                    and hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y \
                    and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
                if len(points):
                    is_drawing = False
                    is_shown = True
                    canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                    canvas_gs = cv2.medianBlur(canvas_gs, 9)
                    canvas_gs = cv2.GaussianBlur(canvas_gs, (5, 5), 0)
                    ys, xs = np.nonzero(canvas_gs)
                    if len(ys) and len(xs):
                        min_y = np.min(ys)
                        max_y = np.max(ys)
                        min_x = np.min(xs)
                        max_x = np.max(xs)
                        cropped_image = canvas_gs[min_y:max_y, min_x: max_x]
                        cropped_image = cv2.resize(cropped_image, (28, 28))
                        cropped_image = np.array(cropped_image, dtype=np.float32)[None, None, :, :]
                        cropped_image = torch.from_numpy(cropped_image)
                        logistic_model = config.model(cropped_image)
                        config.predicted_class = torch.argmax(logistic_model[0])
                        points = collections.deque(maxlen=512)
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                is_drawing = True
                is_shown = False
                points.append((int(hand_landmarks.landmark[8].x * var.new_width),
                               int(hand_landmarks.landmark[8].y * var.new_height)))
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
                    cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 5)
                    utils.draw_landmarks(frame, hand_landmarks)

            if not is_drawing and is_shown:
                cv2.putText(frame, 'You are drawing', (100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5,
                            cv2.LINE_AA)
                frame[5:65, 490:550] = quick_draw_utils.get_overlay(frame[5:65, 490:550],
                                                                    config.class_images[config.predicted_class],
                                                                    (60, 60))
            # step4
            is_horizontal = utils.is_horizontal(frame, results.multi_hand_landmarks)
            # step5
            utils.put_text_horizontal(frame, is_horizontal)
            if is_horizontal and game_clock3 <= 0:
                is_quit = True
            elif is_horizontal:
                game_clock3 -= 1
            else:
                game_clock3 = var.MIN_TIME
            # step6
            utils.add_normal_text_to_right_bottom(frame,
                                                  f"{string_constants.time}: {game_clock3}")
    return points, canvas, is_drawing, is_shown, game_clock3, is_quit
