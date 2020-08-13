#!/usr/bin/env python3

import os
import cv2
import imutils
import shutil
import face_recognition
import numpy as np
import time
from PIL.PngImagePlugin import PngImageFile, PngInfo
import base64
from PIL import Image


class Face():
    key = "face_encoding"

    def __init__(self, face_id, image, face_encoding):
        self.face_id = face_id
        self.image = image
        self.encoding = face_encoding

    def save(self, base_dir):
        # save image
        pathname = os.path.join(base_dir, self.face_id + ".png")
        pil_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        png = Image.fromarray(pil_image)
        b64_str = base64.b64encode(self.encoding.tobytes())
        metadata = PngInfo()
        metadata.add_text(Face.key, b64_str)
        png.save(pathname, pnginfo=metadata)

    @classmethod
    def calculate_encoding(cls, image):
        height, width, channels = image.shape
        top = int(height/3)
        bottom = int(top*2)
        left = int(width/3)
        right = int(left*2)
        box = (top, right, bottom, left)
        return face_recognition.face_encodings(image, [box])[0]

    @classmethod
    def load(cls, pathname):
        # load image
        image = cv2.imread(pathname)
        basename = os.path.basename(pathname)
        stemname, ext = os.path.splitext(basename)
        # load encoding from metadata
        png = PngImageFile(pathname)
        if Face.key in png.info:
            b64_str = png.info[Face.key]
            b64_decoded = base64.b64decode(b64_str)
            face_encoding = np.frombuffer(b64_decoded)
        else:
            face_encoding = Face.calculate_encoding(image)
        new_face = cls(stemname, image, face_encoding)
        return new_face


class Person():
    _last_id = 0

    def __init__(self, name=None):
        if name is None:
            Person._last_id += 1
            self.name = "person_%02d" % Person._last_id
        else:
            self.name = name
            if name.startswith("person_") and name[7:].isdigit():
                id = int(name[7:])
                if id > Person._last_id:
                    Person._last_id = id
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

