import cv2
import numpy as np
import math

from collections import namedtuple

Point = namedtuple("Point", "x, y")

vid = cv2.VideoCapture(0)


# tuning purposes
DEBUG_ALL = True
#DEBUG_SHOW_RAW_CAMERA = False
#DEBUG_RETRO_READING = False
DEBUG_TUNE_RETRO_READING = False
DEBUG_TUNE_BALL_READING = False

# Retro-reflective tape
lower_green = np.array([70,50,120])
upper_green = np.array([100,255,250])

# ball
lower_yellow = np.array([20,50,150])
upper_yellow = np.array([35,180,255])

# home cam
#lower_green = np.array([70,50,125])
#upper_green = np.array([100,255,250])

#lower_yellow = np.array([30,55,100])
#upper_yellow = np.array([50,155,255])

CONSTANT_OF_AREA = 3885

# if about 10% the same
def about_the_same(val_1, val_2):
    avg = math.ceil((val_1 + val_2) / 2)
    return (avg) == (avg) * 0.10


def find_lines(frame, result):
    gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    blur_g = cv2.bilateralFilter(gray, 9, 150, 150)
    canny = cv2.Canny(blur_g, 50, 150, apertureSize = 3)

    im2, contours, h = cv2.findContours(canny, 1, cv2.CHAIN_APPROX_SIMPLE)
    # print("contours", len(contours))
    guidingBoxes = []
    if len(contours) > 0:
        maxContour = max(contours, key=cv2.contourArea)

        # ------------------------------------------------------------------------------------------
        # ROTATE LEFT OR RIGHT OR IS_CENTERED
        min_x = -1
        max_x = -1
        for point in maxContour:
            pt = Point(point[0][0], point[0][1])
            if min_x == -1 or pt.x < min_x:
                min_x = pt.x
            if max_x == -1 or pt.x > max_x:
                max_x = pt.x

        relative_center_x = math.ceil((max_x + min_x)/2)

        width, height = cv2.GetSize(frame)
        screen_center_x = math.ceil((width)/2)

        # if it needs to be further focused
        if not about_the_same(screen_center_x, relative_center_x):
            if relative_center_x > screen_center_x:
                return "right"
            else:
                return "left"

        # ------------------------------------------------------------------------------------------
        # DISTANCE FROM GOAL

        min_y = -1
        max_y = -1
        for point in maxContour:
            pt = Point(point[0][0], point[0][1])
            if min_y == -1 or pt.y < min_y:
                min_y = pt.y
            if max_y == -1 or pt.y > max_y:
                max_y = pt.y

        size = max_y - min_y

        #actual distance  = CONSTANT_OF_AREA / size
        dist = CONSTANT_OF_AREA / size

        speed = get_speed_from_distance(dist)

        # ------------------------------------------------------------------------------------------
        # JUST OUTPUT PROCESSED IMAGE

        #for contour in contours:
        approx = cv2.approxPolyDP(maxContour, 0.10 * cv2.arcLength(maxContour, True), True)
        cv2.drawContours(result, [maxContour], 0, (0, 0, 255), -1)
        cv2.imshow("DEBUG_SHAPE_FINDING", result)


# Testing purposes!!
def on_mouse(k, x, y, s, p):
    global hsv

    if k == 1:   # left mouse, print pixel at x,y
        print(hsv[y, x])

while True:
    _, frame = vid.read()
    # orig: blur = cv2.bilateralFilter(frame, 9, 75, 75)
    blur = cv2.bilateralFilter(frame, 9, 150, 150)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    #mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    #if DEBUG_ALL or DEBUG_TUNE_RETRO_READING or DEBUG_TUNE_BALL_READING:
    #    cv2.namedWindow("DEBUG_COLOUR_TUNING")
    #    cv2.setMouseCallback("DEBUG_COLOUR_TUNING", on_mouse)
    #    cv2.imshow('DEBUG_COLOUR_TUNING', frame)

    #cv2.imshow('Test', result)

    find_lines(frame, result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()