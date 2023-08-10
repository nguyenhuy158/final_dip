import torch

import quick_draw_utils

RED_HSV_LOWER = [0, 100, 100]
RED_HSV_UPPER = [20, 255, 255]
RED_RGB = (0, 0, 255)

GREEN_HSV_LOWER = [36, 0, 0]
GREEN_HSV_UPPER = [86, 255, 255]
GREEN_RGB = (0, 255, 0)

BLUE_HSV_LOWER = [100, 60, 60]
BLUE_HSV_UPPER = [140, 255, 255]
BLUE_RGB = (255, 0, 0)

YELLOW_RGB = (0, 255, 255)
WHITE_RGB = (255, 255, 255)

if torch.cuda.is_available():
    model = torch.load("./trained_models/whole_model_quickdraw")
else:
    model = torch.load("./trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
model.eval()

CLASSES = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer",
           "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]

class_images = quick_draw_utils.get_images("images", CLASSES)

predicted_class = None
