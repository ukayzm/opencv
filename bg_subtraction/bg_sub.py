#

import numpy as np
import cv2

def bg_sub_MOG(cap):
    mog = cv2.bgsegm.createBackgroundSubtractorMOG()

    print('press ESC to quit.')

    while True:
        ret, frame = cap.read()
        fgmask = mog.apply(frame)

        cv2.imshow('mask', fgmask)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

def bg_sub_MOG2(cap):
    mog = cv2.createBackgroundSubtractorMOG2()

    print('press ESC to quit.')

    while True:
        ret, frame = cap.read()
        fgmask = mog.apply(frame)

        cv2.imshow('mask', fgmask)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

def bg_sub_GMG(cap):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    print('press ESC to quit.')

    while True:
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('mask', fgmask)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 320)
bg_sub_MOG(cap)
bg_sub_MOG2(cap)
bg_sub_GMG(cap)
cap.release()
cv2.destroyAllWindows()
