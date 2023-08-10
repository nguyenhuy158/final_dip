import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

paths = ["test{}.jpg".format(i) for i in range(1, 3)]


def loop(path):
    path = "./input/" + path
    print(path)

    # Đọc ảnh
    image = cv2.imread(path, 0)

    # Áp dụng Gaussian Blur để giảm nhiễu
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Dùng Hough Circle Transform để phát hiện các vòng tròn
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=50,
        param2=30,
        minRadius=1,
        maxRadius=20,
    )

    x_start = image.shape[1]
    y_start = image.shape[0]
    x_end = 0
    y_end = 0

    print(x_start, y_start, x_end, y_end)

    # Nếu tìm thấy các vòng tròn
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Vẽ các vòng tròn đã tìm thấy lên ảnh gốc
        for x, y, r in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)

            if x > x_end:
                x_end = x
            if x < x_start:
                x_start = x

            if y > y_end:
                y_end = y
            if y < y_start:
                y_start = y
        print(x_start, y_start, x_end, y_end)
        cv2.rectangle(image, (x_start, y_start),
                      (x_end, y_end), (0, 255, 0), 2)
        print("output/" + path)
        imshow(image)
        cv2.imwrite("output/" + path, image)

        show_graph(circles)
    else:
        print("Không tìm thấy vòng tròn trong ảnh")


def show_graph(circles):
    # Tạo danh sách các tọa độ x và y của các vòng tròn
    x_coords = []
    y_coords = []
    for x, y, r in circles:
        x_coords.append(x)
        y_coords.append(y)

    # Vẽ biểu đồ thống kê vị trí của các vòng tròn
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(x_coords, bins=50)
    axs[0].set_title("Histogram of Circle Positions (X-axis)")
    axs[0].set_xlabel("X-coordinate")
    axs[0].set_ylabel("Frequency")
    axs[1].hist(y_coords, bins=50)
    axs[1].set_title("Histogram of Circle Positions (Y-axis)")
    axs[1].set_xlabel("Y-coordinate")
    axs[1].set_ylabel("Frequency")
    plt.show()


def imshow(image):
    # Hiển thị ảnh gốc và ảnh đã tìm thấy các vòng tròn
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)

    # Thay đổi kích thước cửa sổ hiển thị ảnh
    cv2.resizeWindow("Image", 800, 600)

    # Chờ bấm phím bất kỳ để thoát
    cv2.waitKey(0)

    # Giải phóng bộ nhớ
    cv2.destroyAllWindows()


for path in paths:
    loop(path)
