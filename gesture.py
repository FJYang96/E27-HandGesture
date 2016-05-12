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

def extract(img, mode="Adaptive"):
    """
    Take a gray-scale image and threshold it + apply morphological ops
    """
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
    print 'found', len(contours), 'contours'
    # Find the biggest one
    max_area = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            ci = i
    return contours[i]

def displayContour(contour, image):
    ## Draw Contours
    display = np.zeros((image.shape[0],image.shape[1],3),dtype='uint8')
    cv2.drawContours( display, [contour], 0, [0,255,0], -1)
    return display

cap = cv2.VideoCapture(0)

ret, bg = cap.read()
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg = bg.astype(float)

alpha = 0.95

while( cap.isOpened() ):
    ret, image = cap.read()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = alpha * bg + (1-alpha) * img
    bg_show = bg.astype('uint8')
    fg_show = img - bg_show
    fg_show = extract(fg_show)
    cv2.imshow('win', fg_show)
    while cv2.waitKey(5) < 0: pass
    # TODO:Check ret
    contour = findLargestContour(img)
    display = displayContour(contour, img)
    hull = cv2.convexHull(contour)
    cv2.drawContours( display, [hull], 0, [0,255,0], -1)
