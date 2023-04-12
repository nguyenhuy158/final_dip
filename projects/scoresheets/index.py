from cv2 import *
from os import *


path = 'projects/scoresheets/input/test1.png'
# read img from path
img = imread(path)

# print current working directory
print(getcwd())

# print img shape
print(img.shape)

# show result
imshow('result', img)
waitKey(0)
destroyAllWindows()
