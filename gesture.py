import cv2
import numpy as np

"""
# Read Image
image = cv2.imread('./Marcel-Train/Five/Five-train001.ppm', \
        cv2.IMREAD_GRAYSCALE)

# Show Original Image
cv2.imshow('win',image)
while cv2.waitKey() < 0 : pass
"""
# Background, used for background subtraction(temporal average)
#global bg

def extract(image, mode="Adaptive"):
    """
    Take a gray-scale image and threshold it + apply morphological ops
    """
    img = image.copy()
    if mode == "Adaptive":
        # Use Gaussian adaptive thres
        thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
               cv2.THRESH_BINARY, 11, 2) 
    else:
        # Blur and thres before finding the actual contour
        blur = cv2.GaussianBlur(img, (5,5),0)
        ret, thres = cv2.threshold(blur, 70,255, cv2.THRESH_BINARY_INV)
    # Morph op
    kernel = np.ones( (5,5), np.uint8 )
    opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
    return opening

def findLargestContour(image):
    # Find Contour
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP,\
            cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # Find the biggest one
    max_area = 0
    ci = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            ci = i
    return contours[ci]

def displayContour(contour, image, drawOn=False,color=[0,255,0],filled=-1):
    ## Draw Contours
    display = np.zeros((image.shape[0],image.shape[1],3),dtype='uint8')
    cv2.drawContours( display, [contour], 0, color, filled)
    if not drawOn:
        return display
    else:
        cv2.drawContours( image, [contour], 0, color, filled)
        return display

#def tempAve(bg, frame):
#    """Update bg, return a masked frame with only the foreground"""
#    alpha = 0.7
#    # Find the difference and threshold on the diff
#    diff = np.absolute(frame - bg.astype('uint8'))
#    blur = cv2.GaussianBlur(diff, (5,5),0)
#    _, mask = cv2.threshold(blur, 70,100, cv2.THRESH_BINARY_INV)
#    bmask = mask.view(np.bool)
#    new_frame = np.zeros( (frame.shape[0],frame.shape[1]), 'uint8')
#    new_frame[bmask] = frame[bmask]
#    # Update background
#    bg = alpha * bg + (1-alpha) * frame.astype(float)
#    return new_frame

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=5)

while( cap.isOpened() ):
    ret, image = cap.read()
    img = image.copy()
    fg_mask = fgbg.apply(image)
    contour = findLargestContour(fg_mask)
    if contour is None:
        continue
    display = displayContour(contour,img,drawOn=True)
    hull = cv2.convexHull(contour)
    display = displayContour(hull,img,True,[255,0,0],1)
    cv2.imshow('foreground', display)
    cv2.imshow('video',img)
    cv2.waitKey(5)
