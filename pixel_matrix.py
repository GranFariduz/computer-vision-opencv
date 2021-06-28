import numpy as np
import cv2 as cv

img = np.zeros((10, 5, 3), np.uint8)

# * Pixel Representation * #

# Rows: 10, Columns: 5
# Effectively, Height = Rows, Width = Columns

# Each pixel is black here so:
# assume [0] = [0, 0, 0] in BGR

# Column ------>
# 
# [ [[0][0][0][0][0]],  |
#   [[0][0][0][0][0]],  |
#   [[0][0][0][0][0]],  |
#   [[0][0][0][0][0]],  \/ 
#   [[0][0][0][0][0]],  Row
#   [[0][0][0][0][0]],
#   [[0][0][0][0][0]],
#   [[0][0][0][0][0]],
#   [[0][0][0][0][0]],
#   [[0][0][0][0][0]] ]

# So to access a pixel, we need to use: img[row][column]

# filling the first pixel [0, 0]
# and the last pixel [9, 4] (not [10, 5] as we start from 0) with green
img[0, 0] = [0, 255, 0]
img[9, 4] = [0, 255, 0]

# slicing the matrix, so that we get:
# 2nd and 3rd row, 1st to 3rd column: 
# [ 
#   [[0,0,0], [0,0,0], [0,0,0]], 
#   [[0,0,0], [0,0,0], [0,0,0]] 
# ]

# then we set it to blue color
img[1:3, 0:3] = [255, 0, 0]

cv.imshow('test', img)

cv.waitKey(0)
cv.destroyAllWindows()