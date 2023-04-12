from cv2 import *
from os import *


path = 'projects/scoresheets/input/test1.png'
# read img from path
img = imread(path)

# print current working directory
print(getcwd())

# print img shape
print(img.shape)


img = cvtColor(img, COLOR_BGR2GRAY)
#
retval, dst = threshold(img, 125, 255, THRESH_BINARY)

# show result
imshow('result', dst)
waitKey(0)
destroyAllWindows()
