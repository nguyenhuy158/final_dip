import cv2
import numpy as np

# Load the image
path = 'input/test1.png'
# read img from path
img = cv2.imread(path)

# resize
img = cv2.resize(img, (1024, 720))

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour and check if it has the size and aspect ratio of a score cell
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)
    if area > 100 and area < 500 and aspect_ratio > 0.8 and aspect_ratio < 1.2:
        # This contour is a score cell
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow("Result", img)
cv2.waitKey(0)