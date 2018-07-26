#!/usr/bin/env python3
from __future__ import print_function
import face_recognition
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import os

class Face():
    def __init__(self, frame_id, name, box, encoding):
        self.frame_id = frame_id
        self.name = name
        self.box = box
        self.encoding = encoding

class FaceClustering():
    def __init__(self, src, capture_per_second, skip=0):
        print("%dx%d, %d frame/sec" % (src.get(3), src.get(4), src.get(5)))
        self.skip = skip
        self.faces = []
        self.src = src
        self.frame_id = 0
        self.frame_rate = round(src.get(5))
        self.frames_between_capture = int(self.frame_rate / capture_per_second)
        print("capture every %d frame" % self.frames_between_capture)

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
        if self.frame_id % self.frames_between_capture != 0:
            return True

        if self.frame_id <= self.skip:
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

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self.faces))

    def load(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
            self.faces = pickle.loads(data)

    def getFaceImage(self, image, box):
        img_height, img_width = image.shape[:2]
        (top, right, bottom, left) = box
        box_width = right - left
        box_height = bottom - top
        top = max(top - box_height, 0)
        bottom = min(bottom + box_height, img_height - 1)
        left = max(left - box_width, 0)
        right = min(right + box_width, img_width - 1)
        return image[top:bottom, left:right]

    def cluster(self):
        encodings = [face.encoding for face in self.faces]

        # cluster the embeddings
        print("[INFO] clustering...")
        clt = DBSCAN(metric="euclidean")
        clt.fit(encodings)

        # determine the total number of unique faces found in the dataset
        labelIDs = np.unique(clt.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        print("[INFO] # unique faces: {}".format(numUniqueFaces))

        for labelID in labelIDs:
            # find all indexes into the `data` array that belong to the
            # current label ID, then randomly sample a maximum of 25 indexes
            # from the set
            print("[INFO] faces for face ID: {}".format(labelID))
            indexes = np.where(clt.labels_ == labelID)[0]

            # XXX: mkdir labelID
            os.mkdir("ID%d" % labelID)
            for i in indexes:
                frame_id = self.faces[i].frame_id
                box = self.faces[i].box
                image = cv2.imread("frame_%08d.jpg" % frame_id)
                face_image = self.getFaceImage(image, box)
                cv2.imwrite("ID%d/ID%d-%d.jpg" % (labelID, labelID, i), face_image)

if __name__ == '__main__':
    import argparse
    import pickle

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="video file. Web cam 0 is used when omitted")
    ap.add_argument("-c", "--capture",
                    help="# of frame to capture per second")
    ap.add_argument("-s", "--skip",
                    help="skip the first # of frame")
    args = vars(ap.parse_args())

    if args.get("video", None) is None:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(args["video"])

    if args.get("frame", None) is None:
        capture_per_second = 1
    else:
        capture_per_second = float(args["capture"])

    if args.get("skip", None) is None:
        skip = 0
    else:
        skip = float(args["skip"])

    fc = FaceClustering(cam, capture_per_second, skip)

    #while True:
    #    ret = fc.run()
    #    if ret is False:
    #        break
    #fc.save("encodings.pickle")
    #cam.release()
    #cv2.destroyAllWindows()

    fc.load("encodings.pickle")
    fc.cluster()

    # do a bit of cleanup
    print('finish')
