import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

keyboard = Controller()

#
def get_bbox_coordinates(handLadmark, image_shape):
    """
    Get bounding box coordinates for a hand landmark.
    Args:
        handLadmark: A HandLandmark object.
        image_shape: A tuple of the form (height, width).
    Returns:
        A tuple of the form (xmin, ymin, xmax, ymax).
    """
    all_x, all_y = [], []  # store all x and y points in list
    for hnd in mp_hands.HandLandmark:
        all_x.append(
            int(handLadmark.landmark[hnd].x * image_shape[1])
        )  # multiply x by image width
        all_y.append(
            int(handLadmark.landmark[hnd].y * image_shape[0])
        )  # multiply y by image height

    return (
        min(all_x),
        min(all_y),
        max(all_x),
        max(all_y),
    )  # return as (xmin, ymin, xmax, ymax)


# For static images:
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
            x1, y1, x2, y2 = get_bbox_coordinates(hand_landmarks, image.shape)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 0))
            
            # print(hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)
            if (hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y) \
            and (hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y) \
			and (hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y):
               keyboard.press(Key.space)
            else:
                keyboard.release(Key.space)

    # Flip the image horizontally for a selfie-view display.
    image_display = cv2.flip(image, 1)
    # x1, y1, x2, y2 = get_bbox_coordinates(hand_landmarks, image.shape)
    # image_display = cv2.rectangle(image_display, (x1, y1), (x2, y2))
    cv2.imshow("MediaPipe Hands", image_display)
    if cv2.waitKey(25) & 0xFF == ord("r"):
        break

cap.release()
cv2.destroyAllWindows()
