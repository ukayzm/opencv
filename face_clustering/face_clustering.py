#!/usr/bin/env python3
from __future__ import print_function
import face_recognition
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import os
import pickle
import signal
import sys


class Face():
    def __init__(self, frame_id, name, box, encoding):
        self.frame_id = frame_id
        self.name = name
        self.box = box
        self.encoding = encoding


class FaceClustering():
    def __init__(self):
        self.faces = []
        self.run_encoding = False
        self.capture_dir = "captures"

    def capture_filename(self, frame_id):
        return "frame_%08d.jpg" % frame_id

    def signal_handler(self, sig, frame):
        print(" stop encoding.")
        self.run_encoding = False

    def drawBoxes(self, frame, faces_in_frame):
        # Draw a box around the face
        for face in faces_in_frame:
            (top, right, bottom, left) = face.box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    def encode(self, src_file, capture_per_second, stop=0):
        src = cv2.VideoCapture(src_file)
        if not src.isOpened():
            return

        self.faces = []
        frame_id = 0
        frame_rate = src.get(5)
        stop_at_frame = int(stop * frame_rate)
        frames_between_capture = int(round(frame_rate) / capture_per_second)

        print("start encoding from src: %dx%d, %f frame/sec" % (src.get(3), src.get(4), frame_rate))
        print(" - capture every %d frame" % frames_between_capture)
        if stop_at_frame > 0:
            print(" - stop after %d frame" % stop_at_frame)

        # set SIGINT (^C) handler
        prev_handler = signal.signal(signal.SIGINT, self.signal_handler)
        print("press ^C to stop encoding immediately")

        if not os.path.exists(self.capture_dir):
            os.mkdir(self.capture_dir)

        self.run_encoding = True
        while self.run_encoding:
            ret, frame = src.read()
            if frame is None:
                break

            frame_id += 1
            if frame_id % frames_between_capture != 0:
                continue

            if stop_at_frame > 0 and frame_id > stop_at_frame:
                break

            rgb = frame[:, :, ::-1]
            boxes = face_recognition.face_locations(rgb, model="hog")

            print("frame_id =", frame_id, boxes)
            if not boxes:
                continue

            encodings = face_recognition.face_encodings(rgb, boxes)

            faces_in_frame = []
            for box, encoding in zip(boxes, encodings):
                face = Face(frame_id, None, box, encoding)
                faces_in_frame.append(face)

            # save the frame
            self.drawBoxes(frame, faces_in_frame)
            pathname = os.path.join(self.capture_dir,
                                    self.capture_filename(frame_id))
            cv2.imwrite(pathname, frame)
            #cv2.imshow("Frame", frame)

            self.faces.extend(faces_in_frame)

        # restore SIGINT (^C) handler
        signal.signal(signal.SIGINT, prev_handler)
        self.run_encoding = False
        src.release()
        return

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
        if len(self.faces) is 0:
            print("no faces to cluster")
            return

        print("start clustering %d faces..." % len(self.faces))
        encodings = [face.encoding for face in self.faces]

        # cluster the embeddings
        clt = DBSCAN(metric="euclidean")
        clt.fit(encodings)

        # determine the total number of unique faces found in the dataset
        label_ids = np.unique(clt.labels_)
        num_unique_faces = len(np.where(label_ids > -1)[0])
        print("clustered %d unique faces." % num_unique_faces)

        os.system("rm -rf ID*")
        for label_id in label_ids:
            dir_name = "ID%d" % label_id
            os.mkdir(dir_name)

            # find all indexes of label_id
            indexes = np.where(clt.labels_ == label_id)[0]

            # save face images
            for i in indexes:
                frame_id = self.faces[i].frame_id
                box = self.faces[i].box
                pathname = os.path.join(self.capture_dir,
                                        self.capture_filename(frame_id))
                image = cv2.imread(pathname)
                face_image = self.getFaceImage(image, box)
                filename = dir_name + "-" + self.capture_filename(frame_id)
                pathname = os.path.join(dir_name, filename)
                cv2.imwrite(pathname, face_image)

            print("label_id %d" % label_id, "has %d faces" % len(indexes),
                  "in '%s' directory" % dir_name)

        print('clustering done')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encode",
                    help="video file to encode or '0' to encode web cam")
    ap.add_argument("-c", "--capture", default=1, type=int,
                    help="# of frame to capture per second")
    ap.add_argument("-s", "--stop", default=0, type=int,
                    help="stop encoding after # seconds")
    args = ap.parse_args()

    fc = FaceClustering()

    if args.encode:
        src_file = args.encode
        if src_file == "0":
            src_file = 0
        fc.encode(src_file, args.capture, args.stop)
        fc.save("encodings.pickle")

    try:
        fc.load("encodings.pickle")
    except FileNotFoundError:
        print("No or invalid encoding file. Encode first using -e flag.")
        exit(1)
    fc.cluster()
