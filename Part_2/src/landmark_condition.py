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


def is_horizontal(frame, frame_hands):
    if frame_hands:
        if len(frame_hands) == 1:
            hand_landmarks = frame_hands[0]
            palm_landmarks = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in
                              hand_landmarks.landmark]

            return check_hand_horizontal(palm_landmarks)
    return False


def is_five(hand_landmarks):
    return hand_landmarks.landmark[1].y > hand_landmarks.landmark[4].y and \
        hand_landmarks.landmark[5].y > hand_landmarks.landmark[8].y and \
        hand_landmarks.landmark[9].y > hand_landmarks.landmark[12].y and \
        hand_landmarks.landmark[13].y > hand_landmarks.landmark[16].y and \
        hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y


def is_one(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y and \
        hand_landmarks.landmark[9].y < hand_landmarks.landmark[12].y and \
        hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and \
        hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y


def is_two(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y and \
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y and \
        hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and \
        hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y


def is_three(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y and \
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y and \
        hand_landmarks.landmark[16].y < hand_landmarks.landmark[13].y and \
        hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y
