import cv2
import utils
import string_constants


def dino(frame, game_clock2, is_quit):
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
            else:
                utils.keyboard.release(utils.Key.space)
            # step4
            is_horizontal = utils.is_horizontal(frame, results.multi_hand_landmarks)
            # step5
            utils.put_text_horizontal(frame, is_horizontal)
            if is_horizontal and game_clock2 <= 0:
                is_quit = True
            elif is_horizontal:
                game_clock2 -= 1
            # step6
            utils.add_text_to_image(frame, f"{string_constants.time}: {game_clock2}")

    return game_clock2, is_quit


def rock_paper_scissors(frame, clock, success, game_text, player1, player2, is_quit):
    # Process the frame to detect hands
    results = utils.hands.process(frame)

    # step1
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            utils.draw_landmarks(frame, hand_landmarks)
            player_name = f"P{idx + 1}"
            utils.draw_hand_bounding_box(frame, hand_landmarks, player_name)
    # step2
    clock, success, game_text, player1, player2 = utils.game_running(clock, success,
                                                                     game_text, player1, player2, results)
    # step2.1
    # if not success:
    #     is_horizontal = utils.is_horizontal(frame, results.multi_hand_landmarks)
    #     # step5
    #     utils.put_text_horizontal(frame, is_horizontal)
    # else:
    #     pass

    # step3
    utils.draw_multiline_text(frame, game_text)
    # step4
    utils.add_text_to_image(frame, f"{string_constants.time}: {utils.format_number_lead_zero(clock)}")
    # step5
    clock = (clock + 1) % string_constants.MAX_TIME
    # step6
    return clock, success, game_text, player1, player2, is_quit
