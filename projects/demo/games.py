import utils
import string_constants


def flappy_bird():
    pass


def rock_paper_scissors(frame, clock, success, game_text, player1, player2):
    # Process the frame to detect hands
    results = utils.hands.process(frame)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            utils.draw_landmarks(frame, hand_landmarks)
            player_name = f"P{idx + 1}"
            utils.draw_hand_bounding_box(frame, hand_landmarks, player_name)

    clock, success, game_text, player1, player2 = utils.game_running(clock, success, game_text, player1, player2,
                                                                     results)

    utils.draw_multiline_text(frame, game_text)
    utils.add_text_to_image(frame, f"{string_constants.time}: {utils.format_number_lead_zero(clock)}")

    clock = (clock + 1) % string_constants.MAX_TIME

    return clock, success, game_text, player1, player2
