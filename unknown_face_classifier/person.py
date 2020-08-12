#!/usr/bin/env python3

import os
import cv2
import imutils
import shutil
import face_recognition
import numpy as np
import time
import pickle


class Face():
    _last_id = 0

    def __init__(self, second, image, face_id=None):
        if face_id is None:
            Face._last_id += 1
            self.id = Face._last_id
        else:
            self.id = face_id
            if (Face._last_id < self.id):
                Face._last_id = self.id
        self.second = second
        self.image = image

    def calculate_encoding(self):
        height, width, channels = self.image.shape
        top = int(height/3)
        bottom = int(top*2)
        left = int(width/3)
        right = int(left*2)
        box = (top, right, bottom, left)
        self.encoding = face_recognition.face_encodings(self.image, [box])[0]

    def save(self, base_dir):
        filename = self.second + "-" + str(self.id) + ".png"
        pathname = os.path.join(base_dir, filename)
        cv2.imwrite(pathname, self.image)
        filename = self.second + "-" + str(self.id) + ".pickle"
        pathname = os.path.join(base_dir, filename)
        with open(pathname, "wb") as f:
            pickle.dump(self.encoding, f)

    @classmethod
    def load(cls, pathname):
        image = cv2.imread(pathname)
        basename = os.path.basename(pathname)
        stemname, ext = os.path.splitext(basename)
        splits = stemname.split("-")
        second = splits[0]
        face_id = int(splits[1])
        new_face = cls(second, image, face_id=face_id)
        stempathname, ext = os.path.splitext(pathname)
        filename = stempathname + ".pickle"
        try:
            with open(filename, "rb") as f:
                new_face.encoding = pickle.load(f)
        except OSError as e:
            new_face.calculate_encoding()
        return new_face


class Person():
    _last_id = 0

    def __init__(self, name=None):
        if name is None:
            Person._last_id += 1
            self.name = "person" + str(Person._last_id)
        else:
            self.name = name
        self.encoding = None
        self.faces = []

    def add_face(self, face):
        # add face
        self.faces.append(face)
        # re-calculate encoding
        self.calculate_encoding()

    def calculate_encoding(self):
        if len(self.faces) is 0:
            self.encoding = None
        else:
            encodings = [face.encoding for face in self.faces]
            self.encoding = np.average(encodings, axis=0)

    def distance_statistics(self):
        encodings = [face.encoding for face in self.faces]
        distances = face_recognition.face_distance(encodings, self.encoding)
        return min(distances), np.mean(distances), max(distances)

    def save(self, base_dir):
        pathname = os.path.join(base_dir, self.name)
        try:
            shutil.rmtree(pathname)
        except OSError as e:
            pass
        os.mkdir(pathname)
        for face in self.faces:
            face.save(pathname)

    def save_montages(self, base_dir):
        images = [face.image for face in self.faces]
        montages = imutils.build_montages(images, (128, 128), (6, 2))
        for i, montage in enumerate(montages):
            filename = "montage." + self.name + ("-%02d.png" % i)
            pathname = os.path.join(base_dir, filename)
            cv2.imwrite(pathname, montage)

    @classmethod
    def load(cls, pathname):
        basename = os.path.basename(pathname)
        person = Person(basename)
        for filename in os.listdir(pathname):
            if filename.endswith(".png"):
                face_pathname = os.path.join(pathname, filename)
                face = Face.load(face_pathname)
                person.faces.append(face)
        person.calculate_encoding()
        return person

