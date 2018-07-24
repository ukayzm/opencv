#!/usr/bin/env python3
import face_recognition
import cv2

class FaceClustering():
    def __init__(self):
        self.frame_no = 0
        self.results = []

    def quantify(self, rgb):
        self.frame_no += 1
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        for box, encoding in zip(boxes, encodings):
            result = [self.frame_no, None, box, encoding]
            self.results.append(result)
            print(result)


if __name__ == '__main__':
    import time

    cam = cv2.VideoCapture(0)
    fc = FaceClustering()

    while True:
        ret, frame = cam.read()
        rgb = frame[:, :, ::-1]
        fc.quantify(rgb)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        time.sleep(1)

    print('finish')
