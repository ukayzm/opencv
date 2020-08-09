#!/usr/bin/env python3

class Face():
    def __init__(self, second, image, encoding):
        self.second = second
        self.image = image
        self.encoding = encoding

class Person():
    _id = 1

    def __init__(self, name=None):
        if name is None:
            self.name = "doe" + str(Person._id)
            Person._id += 1
        else:
            self.name = name
        self.directory = None
        self.encoding = None
        self.faces = []
        self.appearances = []

    def add_face(self, face):
        # add face
        self.faces.append(face)
        # re-calculate encoding
        if self.encoding is None:
            self.encoding = face.encoding
        else:
            total_encoding = self.encoding * (len(self.faces) - 1) + face.encoding
            self.encoding = total_encoding / len(self.faces)

