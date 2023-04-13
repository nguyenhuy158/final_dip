import cv2
import os


# windown title
window_title = "result"

# set name of windown
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
    return img_processed


path = "projects/scoresheets/input/test1.png"
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
