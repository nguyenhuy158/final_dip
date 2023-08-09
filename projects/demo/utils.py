import cv2


def draw_multiline_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=1, font_color=(0, 0, 255), thickness=2):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    x, y = position

    lines = text.split('\n')

    for line in lines:
        cv2.putText(image, line, (x, y), font, font_scale, font_color, thickness)
        y += text_size[1] * 1.5
        y = int(y)


