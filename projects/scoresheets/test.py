import cv2
import numpy as np


paths = ['input/test{}.jpg'.format(i) for i in range(1, 16)]

def loop(path):
    print(path)
    
    # Đọc ảnh
    image = cv2.imread(path, 0)

    # Áp dụng Gaussian Blur để giảm nhiễu
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Dùng Hough Circle Transform để phát hiện các vòng tròn
    circles = cv2.HoughCircles(blurred, 
                            cv2.HOUGH_GRADIENT, 
                            1, 
                            20,
                            param1=50, 
                            param2=30, 
                            minRadius=0, 
                            maxRadius=20)

    # Nếu tìm thấy các vòng tròn
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Vẽ các vòng tròn đã tìm thấy lên ảnh gốc
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)

    else:
        print("Không tìm thấy vòng tròn trong ảnh")


def imshow(image):
    # Hiển thị ảnh gốc và ảnh đã tìm thấy các vòng tròn
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', image)

    # Thay đổi kích thước cửa sổ hiển thị ảnh
    cv2.resizeWindow('Image', 800, 600)

    # Chờ bấm phím bất kỳ để thoát
    cv2.waitKey(0)

    # Giải phóng bộ nhớ
    cv2.destroyAllWindows()

    
for path in paths:
    loop(path)