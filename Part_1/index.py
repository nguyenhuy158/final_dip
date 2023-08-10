import cv2
import os
import numpy as np

# window title
window_title = "result"

# set name of window
cv2.namedWindow(window_title)


def on_trackbar(val):
    # function for track bar
    img_processed = process_image(img, val)
    # Show the processed image
    img_processed = cv2.imshow(window_title, img_processed)
    cv2.imshow(window_title, img_processed)


def process_image(img, val):
    # process

    # convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    img_processed = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # kernel = np.ones((3, 3), np.uint8)
    # img_processed = cv2.erode(img_processed, kernel, iterations=1)
    # img_processed = cv2.dilate(img_processed, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        img_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    table_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000 and area < 50000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio > 0.5 and aspect_ratio < 2:
                table_contour = contour
                break

    # Apply perspective transformation to flatten the table
    if table_contour is not None:
        rect = cv2.minAreaRect(table_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        height = max(rect[1])
        width = min(rect[1])
        src_pts = box.astype("float32")
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (int(width), int(height)))

    return warped if warped is not None else img_processed


path = './input/test1.jpg'
# read img from path
img = cv2.imread(path)

# resize img
img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)

# print current working directory
print(os.getcwd())

# print img shape
print(img.shape)

# Create the trackbar
cv2.createTrackbar("parameter", window_title, 0, 255, on_trackbar)

# Show the original image
cv2.imshow(window_title, img)

# Wait for a key press
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()
