import cv2
import numpy as np

def add_text_with_border(image, text):
    # Font và thông số
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # Màu chữ (trắng)
    font_thickness = 2

    # Lấy kích thước của chữ
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Tọa độ để viết chữ (ở giữa hình ảnh)
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2

    # Tạo hình chữ nhật bao quanh chữ
    border_thickness = 10
    border_color = (0, 0, 0)  # Màu đen
    cv2.rectangle(image, (text_x - border_thickness, text_y + border_thickness),
                  (text_x + text_size[0] + border_thickness, text_y - text_size[1] - border_thickness),
                  border_color, -1)  # -1 để vẽ hình chữ nhật đầy đủ

    # Viết chữ lên hình ảnh
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return image

# Đường dẫn tới hình ảnh
image_path = 'woman_hands.jpg'

# Đọc hình ảnh
image = cv2.imread(image_path)

# Chuỗi chữ bạn muốn viết
text = "Hello, OpenCV!"

# Thêm chữ và khung bao quanh chữ lên hình ảnh và lưu lại hình ảnh mới
image_with_text_and_border = add_text_with_border(image.copy(), text)
cv2.imwrite('image_with_text_and_border.jpg', image_with_text_and_border)

# Hiển thị hình ảnh
cv2.imshow('Image with Text and Border', image_with_text_and_border)
cv2.waitKey(0)
cv2.destroyAllWindows()
