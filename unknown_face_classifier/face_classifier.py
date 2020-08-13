#!/usr/bin/env python3

from person import Person
from person import Face
import face_recognition
import os
import shutil
import numpy as np
from datetime import datetime


class FaceClassifier():
    def __init__(self, threshold):
        self.similarity_threshold = threshold
        self.known_persons = []
        self.unknown_dir = "unknowns"
        self.unknowns = Person(self.unknown_dir)

    def get_face_image(self, frame, box):
        img_height, img_width = frame.shape[:2]
        (box_top, box_right, box_bottom, box_left) = box
        box_width = box_right - box_left
        box_height = box_bottom - box_top
        crop_top = max(box_top - box_height, 0)
        pad_top = -min(box_top - box_height, 0)
        crop_bottom = min(box_bottom + box_height, img_height - 1)
        pad_bottom = max(box_bottom + box_height - img_height, 0)
        crop_left = max(box_left - box_width, 0)
        pad_left = -min(box_left - box_width, 0)
        crop_right = min(box_right + box_width, img_width - 1)
        pad_right = max(box_right + box_width - img_width, 0)
        face_image = frame[crop_top:crop_bottom, crop_left:crop_right]
        if (pad_top == 0 and pad_bottom == 0):
            if (pad_left == 0 and pad_right == 0):
                return face_image
        padded = cv2.copyMakeBorder(face_image, pad_top, pad_bottom,
                                    pad_left, pad_right, cv2.BORDER_CONSTANT)
        return padded

    def detect_faces(self, frame):
        faces = []
        rgb = frame[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            return faces

        # faces found
        now = datetime.now()
        str_ms = now.strftime('%Y%m%d_%H%M%S.%f-')[:-3]
        encodings = face_recognition.face_encodings(rgb, boxes)
        for i, box in enumerate(boxes):
            face_image = self.get_face_image(frame, box)
            face = Face(str_ms + str(i), face_image, encodings[i])
            faces.append(face)
        return faces

    def classify_face(self, face):
        # collect encodings of the faces
        known_encodings = [person.encoding for person in self.known_persons]

        if len(known_encodings) > 0:
            # see if the face is a match for the previous faces
            distances = face_recognition.face_distance(known_encodings, face.encoding)
            index = np.argmin(distances)
            min_value = distances[index]
            if min_value < self.similarity_threshold:
                # face of known person
                self.known_persons[index].add_face(face)
                return

        if len(self.unknowns.faces) == 0:
            # this is the first face
            self.unknowns.faces.append(face)
            return

        unknown_encodings = [face.encoding for face in self.unknowns.faces]
        distances = face_recognition.face_distance(unknown_encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.similarity_threshold:
            # two faces are similar
            # create new person with two faces
            person = Person()
            newly_known_face = self.unknowns.faces.pop(index)
            person.add_face(newly_known_face)
            person.add_face(face)
            self.known_persons.append(person)
        else:
            # unknown face
            self.unknowns.faces.append(face)

    def save(self, dir_name):
        print("save persons in the directory", dir_name)
        start_time = time.time()
        try:
            shutil.rmtree(dir_name)
        except OSError as e:
            pass
        os.mkdir(dir_name)
        for person in self.known_persons:
            person.save(dir_name)
            person.save_montages(dir_name)
        self.unknowns.save(dir_name)
        self.unknowns.save_montages(dir_name)
        elapsed_time = time.time() - start_time
        print("save took", elapsed_time, "second")

    def load(self, dir_name):
        print("load persons in the directory", dir_name)
        start_time = time.time()
        for entry in os.scandir(dir_name):
            if entry.is_dir(follow_symlinks=False):
                pathname = os.path.join(dir_name, entry.name)
                person = Person.load(pathname)
                if entry.name == self.unknown_dir:
                    self.unknowns = person
                else:
                    self.known_persons.append(person)
        elapsed_time = time.time() - start_time
        print("load took", elapsed_time, "second")

    def print_persons(self):
        s = "%d persons" % len(self.known_persons)
        s += ", %d known faces" % sum(len(person.faces) for person in
                                      self.known_persons)
        s += ", %d unknown faces" % len(self.unknowns.faces)
        print(s)
        persons = sorted(self.known_persons, key=lambda obj : obj.name)
        encodings = [person.encoding for person in persons]
        for person in persons:
            distances = face_recognition.face_distance(encodings, person.encoding)
            s = "{:10} [ ".format(person.name)
            s += " ".join(["{:5.3f}".format(x) for x in distances])
            mn, av, mx = person.distance_statistics()
            s += " ] %.3f, %.3f, %.3f" % (mn, av, mx)
            s += ", %d faces" % len(person.faces)
            print(s)


if __name__ == '__main__':
    import argparse
    import signal
    import cv2
    import time
    import imutils

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="video file to detect or '0' to detect from web cam")
    ap.add_argument("-c", "--capture", default=1, type=int,
                    help="# of frame to capture per second")
    ap.add_argument("-s", "--stop", default=0, type=int,
                    help="stop detecting after # seconds")
    ap.add_argument("-t", "--threshold", default=0.55, type=float,
                    help="threshold of the similarity")
    args = ap.parse_args()

    src_file = args.input
    if src_file == "0":
        src_file = 0

    src = cv2.VideoCapture(src_file)
    if not src.isOpened():
        print("cannot open input file", src_file)
        exit(1)

    frame_width = src.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = src.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_id = 0
    frame_rate = src.get(5)
    frames_between_capture = int(round(frame_rate) / args.capture)
    #dir_name, ext = os.path.splitext(os.path.basename(args.input))
    dir_name = "result"

    print("source", args.input)
    print("%dx%d, %f frame/sec" % (src.get(3), src.get(4), frame_rate))
    print("capture every %d frame" % frames_between_capture)
    print("similarity shreshold:", args.threshold)
    if args.stop > 0:
        print("will stop after %d seconds." % args.stop)

    running = True

    def signal_handler(sig, frame):
        global running
        running = False

    fc = FaceClassifier(args.threshold)
    if os.path.isdir(dir_name):
        fc.load(dir_name)
        fc.print_persons()

    # set SIGINT (^C) handler
    prev_handler = signal.signal(signal.SIGINT, signal_handler)
    print("press ^C to stop detecting immediately")

    num_new_faces = 0
    while running:
        ret, frame = src.read()
        if frame is None:
            break

        frame_id += 1
        if frame_id % frames_between_capture != 0:
            continue

        seconds = round(frame_id / frame_rate, 3)
        if args.stop > 0 and seconds > args.stop:
            break

        start_time = time.time()

        # this is main
        faces = fc.detect_faces(frame)
        for face in faces:
            fc.classify_face(face)

        num_new_faces += len(faces)
        elapsed_time = time.time() - start_time

        s = "\rframe " + str(frame_id)
        s += " @ time %.3f" % seconds
        s += " takes %.3f seconds" % elapsed_time
        s += ", %d new faces" % len(faces)
        s += " - total %d persons" % len(fc.known_persons)
        s += ", %d known faces" % sum(len(person.faces) for person in
                                      fc.known_persons)
        s += ", %d unknown faces" % len(fc.unknowns.faces)
        print(s, end="    ")

    # restore SIGINT (^C) handler
    signal.signal(signal.SIGINT, prev_handler)
    running = False
    src.release()
    print()

    #if num_new_faces > 0:
    fc.save(dir_name)

    fc.print_persons()
