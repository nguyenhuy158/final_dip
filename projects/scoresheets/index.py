from cv2 import *
from os import *


# windown title
window_title = 'result'

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
    retval, dst = threshold(img, val, 255, THRESH_BINARY)

    return dst


path = 'projects/scoresheets/input/test1.png'
# read img from path
img = imread(path)

# print current working directory
print(getcwd())

# print img shape
print(img.shape)


# Create the trackbar
createTrackbar('parameter',
               window_title,
               0,
               255,
               on_trackbar)


# Show the original image
imshow(window_title, img)

# Wait for a key press
waitKey(0)

# Destroy all windows
destroyAllWindows()
