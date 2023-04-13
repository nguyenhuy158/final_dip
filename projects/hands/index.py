from cv2 import *
from os import *


# windown title
window_title = "result"

# set name of windown
namedWindow(window_title)


def on_trackbar(val):
    # function for track bar
    img_processed = process_image(img, val)
    # Show the processed image
    img_processed = imshow(window_title, img_processed)
    imshow(window_title, img_processed)


def process_image(img, val):
    # process

    # convert to gray
    img = cvtColor(img, COLOR_BGR2GRAY)

    # threshold
    img_processed = adaptiveThreshold(
        img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2
    )
    return img_processed


path = "projects/scoresheets/input/test1.png"
# read img from path
img = imread(path)

# resize img
img = resize(img, (0, 0), fx=0.7, fy=0.7)


# print current working directory
print(getcwd())

# print img shape
print(img.shape)


# Create the trackbar
createTrackbar("parameter", window_title, 0, 255, on_trackbar)


# Show the original image
imshow(window_title, img)

# Wait for a key press
waitKey(0)

# Destroy all windows
destroyAllWindows()
