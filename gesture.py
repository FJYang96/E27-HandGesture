import cv2
import numpy as np
from numpy.linalg import det
from math import sqrt
from sys import argv

global palm_centers

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
    return thres

def findLargestContour(image):
    # Find Contour
    #RETR_EXTERNAL only considers the outermost contour, CHAIN_APPROX_SIMPLE approximates the contour
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    
def handAnalysis(image, contour, hull):
    """
    Analyze the gesture of the hand in the image using contour and hull
    """
    # Find defect points
    defects = cv2.convexityDefects(contour,hull)
    # Unpack the points in defects and find the average defect
    defs = []
    defects_new = []
    cent = np.zeros( (2), float)
    if len(defects) > 3:
        for defect in defects:
            # start and end on the hull, far is the valley
            start = tuple(contour[defect[0,0]][0])
            end = tuple(contour[defect[0,1]][0])
            far = tuple(contour[defect[0,2]][0])
            depth = defect[0,3]
            # Threshold on depth, filter out the defect points near the hull
            if depth > 1500:
                print depth
                cent = cent + far
                defs.append(far)
                defs.append(start)
                defs.append(end)
                defects_new.append(defect)
                cv2.circle(image, far, 3, (0,0,255), -1)

        # Find the initial estimation of the palm center
        centx = cent[0] / (len(defs)*3)
        centy = cent[1] / (len(defs)*3)
        sorted(defs, key=lambda p: sqrt((p[0]-centx)**2 + (p[1]-centy)**2))

        cent, radius = circleFromPoints(defs[0], defs[1], defs[2])

        palm_centers.append( (cent, radius) )

        centx, centy, radius = 0, 0, 0

        if len(palm_centers) > 10:
            length = 11
        else:
            length = len(palm_centers)

        for i in range(1, length):
            centx += palm_centers[-i][0][0]
            centy += palm_centers[-i][0][1]
            radius += palm_centers[-i][1]

        centx = centx / length
        centy = centy / length
        radius = radius / length

        # Draw the center of circle and circle itself in the image
        cv2.circle(image, (int(centx), int(centy)), 5, (255,255,255), 2)
        cv2.circle(image, (int(centx), int(centy)), int(radius), (255,255,255), 2)

        # Detect finger
        numFingers = 0
        palm_center = (centx, centy)
        for defect in defects_new:
            # start and end on the hull, far is the valley
            start = tuple(contour[defect[0,0]][0])
            end = tuple(contour[defect[0,1]][0])
            far = tuple(contour[defect[0,2]][0])
            Xdist = dist(palm_center, far)
            Ydist = dist(palm_center, start)
            length = dist(far, start)
            retLength = dist(end, far)
            if (length<=3*radius and Ydist>=0.4*radius and length>=10 and retLength>=10 and max(length,retLength)/min(length,retLength)>=0.8):
                if(min(Xdist,Ydist)/max(Xdist,Ydist)<=0.8):
                    if((Xdist>=0.1*radius and Xdist<=1.3*radius and Xdist<Ydist) or (Ydist>=0.1*radius and Ydist<=1.3*radius and Xdist>Ydist)):
                        cv2.line(image, end, far, (255, 255, 255), 2)
                        numFingers += 1
            

def dist(p1, p2):
    """
    Euclidean distance metric
    """
    d1 = p1[0] - p2[0]
    d2 = p1[1] - p2[1]
    return sqrt(d1**2 + d2**2)

def valleyEndTest(start, end, far):
    distStart = dist(start, far)
    distEnd = dist(end, far)
    if abs(distStart-distEnd) < 5:
        return True
    else:
        return False

def circleFromPoints(p1, p2, p3):
    """
    This function takes three points in the image and returns the center and radius of the circle passing through these 
    three points
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    centerx = det(np.array(([x1**2+y1**2,y1,1],[x2**2+y2**2,y2,1],[x3**2+y3**2,y3,1]))) / float(2*det(np.array(([x1,y1,1],[x2,y2,1],[x3,y3,1]))))
    centery = det(np.array(([x1,x1**2+y1**2,1],[x2,x2**2+y2**2,1],[x3,x3**2+y3**2,1]))) / float(2*det(np.array(([x1,y1,1],[x2,y2,1],[x3,y3,1]))))
    radius = sqrt( pow(x2 - centerx, 2) + pow(y2-centery, 2))
    # Alternative:
    # offset = pow(x2, 2) +pow(y2, 2)
    # bc = ( pow(x1, 2) + pow(y1, 2) - offset )/2.0
    # cd = (offset - pow(x3, 2) - pow(y3, 2))/2.0
    # det = (x1 - x2) * (y2 - y3) - (x2 - x3)* (y1 - y2) 
    # TOL = 0.0000001
    # if (abs(det) < TOL):
    #     print "POINTS TOO CLOSE"
    #     return ((0, 0), 0) 
    # idet = 1/det
    # centerx =  (bc * (y2 - y3) - cd * (y1 - y2)) * idet
    # centery =  (cd * (x1 - x2) - bc * (x2 - x3)) * idet
    # print centerx, centery, radius
    return ((centerx, centery), radius)


########## Main ##########
camera = True

if len(argv) != 2:
    print "Cannot parse input\nUsage: python gesture.py [filename]"
    print "A filename of 0 means running it on a camera device"
    exit(1)
    
# You can specify a video device (e.g. 0) on the command line
# or a static image filename
try:
    device_num = int(argv[1])
    cap = cv2.VideoCapture(device_num)
except:
    src_file = argv[1]
    camera = False

# Static Image
if not camera:
    # Read Image
    image = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
    if image is None or not len(image):
        print "Image not available, exit"
        exit(1)
    print "Running on input image"
    img = extract(image)
    contour = findLargestContour(img)
    # Find the indices of points in the contour array that are on the hull
    hull = cv2.convexHull(contour,returnPoints=False)
    display = displayInfo(image, contour, hull)
    handAnalysis(display, contour, hull)
    # Show Original Image
    cv2.imshow('win', display)
    while cv2.waitKey() < 0 : pass

# Camera Mode
if camera:
    print "Running on camera mode"
    # Create a background subtractor  using MOG2, not detecting shadow
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=5,detectShadows=False)

    palm_centers = []

    # Main while loop for camera
    while( cap.isOpened() ):
        ret, image = cap.read()
        if not ret or image is None:
            break
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
        # Draw the contour using green, the hull polygon using blue
        display = displayInfo(img, contour, hull)

        # Find and filter the defect points and draw them in red
        handAnalysis(display, contour, hull)

        cv2.imshow('video', display)
        while cv2.waitKey(1) < 0: pass 
        #k = cv2.waitKey(10)
        #if k == 27:
        #    break


##### Temperal Averaging Implementation #####
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
