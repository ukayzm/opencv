# image_detector.py

from object_detector import *
import cv2
import numpy as np
import sys

def usage():
    print("usage:", sys.argv[0], "img_file")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("no image file given.")
        usage()
        exit()

    #model = 'pet'
    model = 'ssd_mobilenet_v1_coco_2017_11_17'
    #model = 'mask_rcnn_inception_v2_coco_2018_01_28'

    print("ObjectDetector('%s')" % model)
    detector = ObjectDetector(model)

    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    print("image file:", sys.argv[1], "(%dx%d)" % (width, height))

    print("press any key to quit")
    frame = detector.detect_objects(img)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)

    # do a bit of cleanup
    cv2.destroyAllWindows()
