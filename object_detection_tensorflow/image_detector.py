# image_detector.py

from object_detector import *
import cv2
import numpy as np
import sys
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output-file', default=None,
                    help='detected frame is saved to file')
    ap.add_argument('image_file', help='image_file_to_run_inference')
    args = ap.parse_args()
    print(args)

    #detector = ObjectDetector('ssd_mobilenet_v1_coco_2017_11_17')
    #detector = ObjectDetector('mask_rcnn_inception_v2_coco_2018_01_28')
    detector = ObjectDetector('pet', label_file='data/pet_label_map.pbtxt')

    img = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    print("image file:", args.image_file, "(%dx%d)" % (width, height))

    frame = detector.detect_objects(img)

    if args.output_file:
        cv2.imwrite(args.output_file, frame)
    else:
        # show the frame
        print("press any key to quit")
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0)

        # do a bit of cleanup
        cv2.destroyAllWindows()
