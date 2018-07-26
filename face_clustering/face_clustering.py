#!/usr/bin/env python3
from __future__ import print_function
import face_recognition
import cv2

class Face():
    def __init__(self, frame_id, name, box, encoding):
        self.frame_id = frame_id
        self.name = name
        self.box = box
        self.encoding = encoding

class FaceClustering():
    def __init__(self, src, frame_per_second):
        print("%dx%d, %d frame/sec" % (src.get(3), src.get(4), src.get(5)))
        self.faces = []
        self.src = src
        self.frame_id = 0
        self.frame_rate = round(src.get(5))
        self.capture_per_frame = int(self.frame_rate / frame_per_second)
        print("capture every %d frame" % self.capture_per_frame)

    def quantify(self, frame_id, rgb):
        faces_in_frame = []
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        for box, encoding in zip(boxes, encodings):
            face = Face(frame_id, None, box, encoding)
            faces_in_frame.append(face)
        return faces_in_frame

    def drawBox(self, frame, faces_in_frame):
        # Draw a box around the face
        for face in faces_in_frame:
            (top, right, bottom, left) = face.box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    def run(self):
        if self.src.isOpened() is False:
            return False

        ret, frame = self.src.read()
        if frame is None:
            return False

        self.frame_id += 1
        if self.frame_id % self.capture_per_frame != 0:
            return True

        rgb = frame[:, :, ::-1]
        faces_in_frame = self.quantify(self.frame_id, rgb)

        print("frame_id =", self.frame_id, faces_in_frame)

        if not faces_in_frame:
            return True

        # show the frame
        self.drawBox(frame, faces_in_frame)
        cv2.imwrite("frame_%08d.jpg" % self.frame_id, frame)
        #cv2.imshow("Frame", frame)

        self.faces.extend(faces_in_frame)

        return True


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="video file. Web cam 0 is used when omitted")
    ap.add_argument("-f", "--frame",
                    help="# of frame to capture per second")
    args = vars(ap.parse_args())

    if args.get("video", None) is None:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(args["video"])

    if args.get("frame", None) is None:
        frame_per_second = 1
    else:
        frame_per_second = float(args["frame"])

    fc = FaceClustering(cam, frame_per_second)

    while True:
        ret = fc.run()
        if ret is False:
            break

    # do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()
    print('finish')
