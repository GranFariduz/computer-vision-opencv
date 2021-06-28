import datetime
import cv2 as cv
import numpy as np

# * READ AN IMAGE * #
# flags: 1, 0, -1
#   1: Loads color image
#   0: Loads image in grayscale
#   -1: Loads image as such with alpha channel
# img = cv.imread('samples/lena.jpg', 0)

# * SHOW AN IMAGE * #
# cv.imshow('OpenCV', img)
# waits for given milliseconds before closing imshow window
# using 0 will wait until user closes window
# key = cv.waitKey(0)

# * WRITE AN IMAGE FILE * #
# cv.imwrite('generated/lena-gray.png', img)

# if esc is pressed, close all windows
# else write the file and close all windows
# if key == 27:
#   # destroys all open windows (from opencv)
#   cv.destroyAllWindows()
# elif key == ord('s'):
#   cv.imwrite('generated/lena-gray.png', img)
#   cv.destroyAllWindows()

# * DRAW GEOMETRIC SHAPES * #
# img = cv.imread('samples/loki.jpg', 1)

# cv.line(img, (0, 0), (200, 200), (240, 100, 98), 2)
# cv.arrowedLine(img, (0, 200), (200, 200), (0, 255, 0), 2)

# cv.rectangle(img, (0, 300), (300, 100), (0, 0, 255), 3, cv.LINE_AA)
# if we put thickness as -1, it fills the circle
# cv.circle(img, (200, 200), 50, (200, 0, 0), 2, cv.LINE_AA) 

# cv.putText(img, 'Loki', (10, 490), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (17, 142, 33), 2, cv.LINE_AA)

# cv.imshow('Loki', img)

# cv.waitKey(0)
# cv.destroyAllWindows()

# * VIDEO CAPTURE * #
# # can also use a string to the path of a video
# video_cap = cv.VideoCapture(0)

# # getting additonal information
# print(video_cap.get(cv.CAP_PROP_FRAME_WIDTH))

# # setting properties to video
# # will use the closest available resolution value possible
# video_cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# video_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# # defining the four digit video codec, creating video writer
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# vid_writer = cv.VideoWriter('generated/sample_vid.avi', fourcc, 20.0, (1024, 720))

# if not video_cap.isOpened():
#   print("Cannot read camera")
#   exit()
# while True:
#   # we need a loop because we only get a frame from .read()
#   # the loop runs the frames, making it seem like a video 
  
#   # frame returns the next frame data (a list)
#   # ret returns if the frame is read correctly
#   ret, frame = video_cap.read()
  
#   if not ret:
#     print('Cannot receive frames, exiting...')
#     break
#   else:
#     # writing the vertically flipped video to a file
#     flipped_frame = cv.flip(frame, 0)
#     vid_writer.write(flipped_frame)

#     # by default, the channels are BGR not RGB
#     gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # adding text to the video
#     cv.putText(
#       gray_frame, 
#       f"Width: {video_cap.get(cv.CAP_PROP_FRAME_WIDTH)}, Height: {video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)}",
#       (10, 35),
#       cv.FONT_HERSHEY_SIMPLEX,
#       1,
#       (0, 0, 0),
#       2,
#       cv.LINE_AA
#     )

#     cv.putText(
#       gray_frame, 
#       f"{datetime.datetime.now()}",
#       (10, 90),
#       cv.FONT_HERSHEY_SIMPLEX,
#       1,
#       (0, 0, 0),
#       2,
#       cv.LINE_AA
#     )

#     cv.imshow('VideoCapture', gray_frame)

#     # wait for 1 ms to press a key
#     # the window never closes because it's an infinite loop
#     if cv.waitKey(1) == ord('q'):
#       break

# # closes capturing device or video file
# video_cap.release()
# cv.destroyAllWindows()

# * MOUSE EVENTS * #
# def click_event(event, x, y, flags, param):
#   if event == cv.EVENT_LBUTTONDOWN:
#     cv.putText(img, f"{x}, {y}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
#   elif event == cv.EVENT_RBUTTONDOWN:
#     # to get a channel color from within the img,
#     # we need to pass the coords and the channel index
#     # 0: Blue, 1: Green, 2: Red

#     # ? Read: pixel_matrix.py
#     # you can access a pixel by img[row, column, channel-index]
#     # that is why its img[y, x] and not img[x, y]
#     # For Images (which are basically matrices), OpenCV uses matrix notation i.e [row][column]
#     # For Points, OpenCV uses cartesian notation i.e [x, y]
#     pixel_bgr_values = [img[y, x, 0], img[y, x, 1], img[y, x, 2]]
#     cv.putText(img, f"{pixel_bgr_values[0]}, {pixel_bgr_values[1]}, {pixel_bgr_values[2]}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
#   elif event == cv.EVENT_MBUTTONDOWN:
#     cv.circle(img, (x, y), 2, (0, 255, 0), -1, cv.LINE_AA)

#     points.append((x, y))

#     if len(points) >= 2:
#       # getting last 2 points and connecting them using a line
#       cv.line(img, points[-1], points[-2], (0, 255, 0), 1, cv.LINE_AA)
#   elif event == cv.EVENT_LBUTTONDBLCLK:
#     pixel_bgr_values = [img[y, x, 0], img[y, x, 1], img[y, x, 2]]

#     color_image = np.zeros((256, 350, 3), np.uint8)
#     color_image[:] = pixel_bgr_values

#     cv.imshow(f'Color Picker rgb({pixel_bgr_values[0]}, {pixel_bgr_values[1]}, {pixel_bgr_values[2]})', color_image)

# # np.zeros shape: (rows, columns, no of channels [should be 3 for color, and no value for grayscale])
# # img = np.zeros((512, 512, 3), np.uint8)
# img = cv.imread('samples/messi5.jpg', 1)

# points = []

# # making a placeholder window and binding the the callback to it
# cv.namedWindow('image')
# cv.setMouseCallback('image', click_event)

# # to refresh the img
# while(1):
#   cv.imshow('image', img)
  
#   if cv.waitKey(20) == ord('q'):
#     break

# cv.destroyAllWindows()

# * COPYING PARTS (EX: REGION OF INTEREST) * #
# img = cv.imread('samples/messi5.jpg', 1)

# # we need two opposite vertices to form a rectangle
# # x1, y1
# # [63, 81] x2, y2
# #         [118, 134]

# # to get the part of an image using the two vertices:
# # img[y1:y2, x1:x2]
# # it will slice the matrix from y1 to y2, and x1 to x2
# hand = img[81:134, 63:118]
# # copying the hand to 100px down in y direction
# img[181:234, 63:118] = hand

# cv.imshow('Messi', img)

# cv.waitKey(0)
# cv.destroyAllWindows()

# * BLENDING TWO IMAGES * #
# img = cv.imread('samples/messi5.jpg', 1)
# img2 = cv.imread('samples/opencv-logo.png', 1)

# # blending won't work until they are of the same sizes
# img = cv.resize(img, (512, 512))
# img2 = cv.resize(img2, (512, 512))

# res = cv.add(img, img2)

# # blend with specific weights
# resWeighted = cv.addWeighted(img, 0.8, img2, 0.2, 0)

# cv.imshow('Blend', resWeighted)

# cv.waitKey(0)
# cv.destroyAllWindows()

# * BITWISE OPERATIONS * #
# # Both images should have exact same dimensions (i.e Rows and columns)
# img1 = np.zeros((250, 499, 3), np.uint8)
# cv.rectangle(img1, (200, 0), (300, 100), (255, 255, 255), -1)

# img2 = cv.imread('samples/1bit1.png', 1)

# # ? check truth tables of AND, OR, XOR, NOT
# bitAND = cv.bitwise_and(img1, img2)
# bitOR = cv.bitwise_or(img1, img2)
# bitXOR = cv.bitwise_xor(img1, img2)
# bitNOT = cv.bitwise_not(img1) # just gives you the opposite

# cv.imshow('img1', img1)
# cv.imshow('img2', img2)

# cv.imshow('BIT_AND', bitAND)
# cv.imshow('BIT_OR', bitOR)
# cv.imshow('BIT_XOR', bitXOR)
# cv.imshow('BIT_NOT', bitNOT)

# cv.waitKey(0)
# cv.destroyAllWindows()

# * ADDING TRACKBARS TO WINDOWS * #
# img = np.zeros((300, 300, 3), np.uint8)

# window_name = 'RGB Color Range'
# switch_name = 'OFF / ON'

# cv.namedWindow(window_name)

# def on_trackbar_change(val):
#   pass

# cv.createTrackbar('B', window_name, 0, 255, on_trackbar_change)
# cv.createTrackbar('G', window_name, 0, 255, on_trackbar_change)
# cv.createTrackbar('R', window_name, 0, 255, on_trackbar_change)
# cv.createTrackbar(switch_name, window_name, 0, 1, on_trackbar_change)

# while(1):
#   cv.imshow(window_name, img)

#   if cv.waitKey(1) == 27:
#     break

#   b = cv.getTrackbarPos('B', window_name)
#   g = cv.getTrackbarPos('G', window_name)
#   r = cv.getTrackbarPos('R', window_name)
#   switch_val = cv.getTrackbarPos(switch_name, window_name)

#   if switch_val == 0:
#     img[:] = 0 # make every pixel black if switched off
#   else:
#     img[:] = [b, g, r]

# cv.destroyAllWindows()

# * CHANGING COLOR MODES USING TRACKBAR (CONTINUATION) * #
# window_name = 'Color Mode'

# cv.namedWindow(window_name)

# cv.createTrackbar('Mode', window_name, 0, 1, lambda a: None)

# while(1):
#   img = cv.imread('samples/lena.jpg')
#   mode = cv.getTrackbarPos('Mode', window_name)

#   if mode == 0:
#     pass
#   else:
#     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#   cv.imshow(window_name, img)

#   if cv.waitKey(1) == 27:
#     break

# cv.destroyAllWindows()

# * DETECTING OBJECTS THROUGH UPPER AND LOWER HSV VALUES * #
# # For detecting objects, we have to convert the image into HSV color space
# # Why? Because the BGR color space is related to luminance (or the amount of light hitting object)
# # So differentiating color becomes difficult, as we cannot separate color from luminance
# # In HSV, we have a clear distinction between the color (Hue), and the Saturation and brightness

# # Hue: The pure pigment, without tint or shade (added white or black pigment)
# # Saturation: The amount to which a color is towards gray
# # Value: Basically the brightness

# # The simple answer is that unlike RGB, 
# # HSV separates luma, or the image intensity, from chroma or the color information. 

# # for HSV (in OpenCV),
# # Hue range is [0, 179]
# # Saturation range is [0, 255]
# # Value range is [0, 255]

# window_name = 'tracking'

# cv.namedWindow(window_name)

# def nothing(val):
#   pass

# cv.createTrackbar('LH', window_name, 0, 179, nothing)
# cv.createTrackbar('LS', window_name, 0, 255, nothing)
# cv.createTrackbar('LV', window_name, 0, 255, nothing)
# cv.createTrackbar('UH', window_name, 179, 179, nothing)
# cv.createTrackbar('US', window_name, 255, 255, nothing)
# cv.createTrackbar('UV', window_name, 255, 255, nothing)

# video_cap = cv.VideoCapture(0)

# while(1):
#   # img = cv.imread('samples/messi5.jpg')
#   ret, img = video_cap.read()
#   hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#   lh = cv.getTrackbarPos('LH', window_name)
#   ls = cv.getTrackbarPos('LS', window_name)
#   lv = cv.getTrackbarPos('LV', window_name)

#   uh = cv.getTrackbarPos('UH', window_name)
#   us = cv.getTrackbarPos('US', window_name)
#   uv = cv.getTrackbarPos('UV', window_name)

#   # we need a numpy array for defining boundaries
#   # the lower boundary and upper boundary are just
#   # lower and upper limits for the color you want to detect
#   upper_boundary = np.array([lh, ls, lv], np.uint8)
#   lower_boundary = np.array([uh, us, uv], np.uint8)

#   # After calling cv2.inRange, a binary mask is returned, 
#   # where white pixels (255) represent pixels that fall into the upper and lower limit 
#   # and black pixels (0) do not
#   mask = cv.inRange(hsv_image, lower_boundary, upper_boundary)

#   # Only in this situation, i.e bitwise AND
#   # white parts of the mask means what parts to keep
#   # but black parts mean which parts to remove
#   # result will give us only the white parts corresponding to the image
#   res = cv.bitwise_and(img, img, mask=mask)

#   cv.imshow('image', img)
#   cv.imshow('mask', mask)
#   cv.imshow('result', res)

#   if cv.waitKey(1) == 27:
#     break

# cv.destroyAllWindows()

# * SIMPLE THRESHOLDING IN IMAGES * #
# # Thresholding is a technique in OpenCV, which is the assignment of pixel values 
# # in relation to the threshold value provided. 
# # In thresholding, each pixel value is compared with the threshold value.

# # IN GRAYSCALE
# # Since grayscale images have only one value i.e between 0 to 255,
# # as opposed to [B, G, R] in color images, it will only give us a result
# # in 0 (black) or (255) white per pixel
# img = cv.imread('samples/lena.jpg', 1)

# # values lesser than 50 will have a value of 0
# # values greater than 50 will have a value of 255
# _, binary_th = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

# # opposite of THRESH_BINARY
# _, binary_inv_th = cv.threshold(img, 50, 255, cv.THRESH_BINARY_INV)

# # values lesser than 120 will be unchnaged
# # values greater than 120 will have a value of 120
# _, trunc_th = cv.threshold(img, 120, 255, cv.THRESH_TRUNC)

# # values lesser than 120 will have a value of 0
# # values greater than 120 will be unchanged
# _, tozero_th = cv.threshold(img, 120, 255, cv.THRESH_TOZERO)

# # opposite of THRESH_TOZERO
# _, tozero_inv_th = cv.threshold(img, 120, 255, cv.THRESH_TOZERO_INV)

# # IN COLOR
# # Since each pixel will be assigned a value according to the threshold,
# # this means, a 0 or 255 value will be applied to all the values in BGR,
# # making the image seem like a mishmash of colors

# cv.imshow('Thresholding', binary_th)

# cv.waitKey(0)
# cv.destroyAllWindows()

# * ADAPTIVE THRESHOLDING IN IMAGES * #
# ? Must Read: https://www.pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/

# In basic thresholding and Otsu's thresholding, they apply global thresholding,
# implying that the same threshold value is used to test all pixels in the input image, 
# thereby segmenting them into foreground and background

# The problem here is that having just one value of T may not suffice. 
# Due to variations in lighting conditions, shadowing, etc., 
# it may be that one value of T will work for a certain part of the input image 
# but will utterly fail on a different segment.

# Adaptive thresholding considers a small set of neighboring pixels at a time, 
# computes threshold value for that specific local region, and then performs the segmentation.

# Adaptive thresholding only works for grayscale images
img = cv.imread('samples/sudoku.png', 0)

# threshold value = (mean of local subregion of image (ex: 20x20 block)) – some constant value we can use to fine tune the threshold value

mean_th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 13, 15)
gauss_th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 15)

cv.imshow('original', img)
cv.imshow('image', gauss_th)

cv.waitKey(0)
cv.destroyAllWindows()
