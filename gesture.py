import cv2
import numpy as np
from numpy.linalg import det
from math import sqrt
from sys import argv

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

def displayInfo(image, contour, hull):
    """
    Draw the contour and hull of the contour on the image
    Here hull is the 1D indices matrix
    """
    #display = np.zeros((image.shape[0],image.shape[1],3),dtype='uint8')
    img = image.copy()
    cv2.drawContours( img, [contour], 0, [0,255,0], -1)
    hull_points = np.array( [contour[i][0] for i in hull] )
    cv2.drawContours( img, [hull_points], 0, [255,0,0], 3)
    return img

def findCenter(points, center_estimate=None):
    """
    Given a set of defects, find the approximate center of plam
    first find average
    then iteratively find the center of the circle that passes through
    the three closest points to current center
    """
    # First find the average
    cent = np.zeros( (2), float)
    if center_estimate is not None:
        cent = center_estimate
    else:
        for point in points:
            cent += point
        cent = cent / len(points)

    if len(points) < 3:
        return (int(cent[0]), int(cent[1]))

    return (int(cent[0]), int(cent[1]))
    for i in range(3):
        # Find the three closest points
        sorted(points,\
                key=lambda p: sqrt((p[0]-cent[0])**2 + (p[1]-cent[1])**2))
        # Find the center of circle, update cent
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        # Hard coded equation
        #cent[0] = det(np.array(([x1**2+y1**2,y1,1],[x2**2+y2**2,y2,1],[x3**2+y3**2,y3,1]))) / float(2*det(np.array(([x1,y1,1],[x2,y2,1],[x3,y3,1]))))
        #cent[1] = det(np.array(([x1,x1**2+y1**2,1],[x2,x2**2+y2**2,1],[x3,x3**2+y3**2,1]))) / float(2*det(np.array(([x1,y1,1],[x2,y2,1],[x3,y3,1]))))

    print cent
    return (int(cent[0]), int(cent[1]))
    
def handAnalysis(image, contour, hull):
    """
    Analyze the gesture of the hand in the image using contour and hull
    """
    # Find defect points
    defects = cv2.convexityDefects(contour,hull)
    # Unpack the points in defects and find the average defect
    defs = []
    for defect in defects:
        # start and end on the hull, far is the valley
        start = tuple(contour[defect[0,0]][0])
        end = tuple(contour[defect[0,1]][0])
        far = tuple(contour[defect[0,2]][0])
        depth = defect[0,3]
        defs.append(far)
        cv2.circle(image, far, 3, (255,255,0), -1)
    # Find the center of the palm
    #cent = findCenter(defs)
    #cv2.circle(image, cent, 5, (255,255,0), 2)
    # Iteratively find the three closest points in def to current center
     
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

camera = False

if len(argv) > 2:
    print "Cannot parse input\nUsage: python gesture.py [filename]"
    print "Running the program without a filename argument means\
            running it on a camera device"
    exit(1)
elif len(argv) == 1:
    print "Running on camera mode"
    camera = True
else:
    print "Running on input image"

if not camera:
    # Read Image
    image = cv2.imread('./Marcel-Train/Five/Five-train001.ppm', \
            cv2.IMREAD_GRAYSCALE)
    img = extract(image)
    contour = findLargestContour(img)
    # Find the indices of points in the contour array that are on the hull
    hull = cv2.convexHull(contour,returnPoints=False)
    display = displayInfo(image, contour, hull)
    handAnalysis(display, contour, hull)
    # Show Original Image
    cv2.imshow('win',display)
    while cv2.waitKey() < 0 : pass

if camera:
    cap = cv2.VideoCapture(0)

    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=5)

    while( cap.isOpened() ):
        ret, image = cap.read()
        # Make a copy of the frame
        img = image.copy()
        # Background subtraction
        fg_mask = fgbg.apply(image)
        # Find the largest contour, if cannot find any skip this frame
        contour = findLargestContour(fg_mask)
        if contour is None:
            continue
        # Find the indices of points on the contour that are on the hull
        hull = cv2.convexHull(contour, returnPoints=False)
        display = displayInfo(img, contour, hull)
        handAnalysis(display, contour, hull)
        cv2.imshow('video',display)
        while cv2.waitKey(5) < 0: pass
