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
    key = "face_encoding"

    def __init__(self, filename, image, face_encoding):
        self.filename = filename
        self.image = image
        self.encoding = face_encoding

    def save(self, base_dir):
        # save image
        pathname = os.path.join(base_dir, self.filename)
        cv2.imwrite(pathname, self.image)

    @classmethod
    def get_encoding(cls, image):
        rgb = image[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            height, width, channels = image.shape
            top = int(height/3)
            bottom = int(top*2)
            left = int(width/3)
            right = int(left*2)
            box = (top, right, bottom, left)
        else:
            box = boxes[0]
        return face_recognition.face_encodings(image, [box])[0]


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
        self.calculate_average_encoding()

    def calculate_average_encoding(self):
        if len(self.faces) is 0:
            self.encoding = None
        else:
            encodings = [face.encoding for face in self.faces]
            self.encoding = np.average(encodings, axis=0)

    def distance_statistics(self):
        encodings = [face.encoding for face in self.faces]
        distances = face_recognition.face_distance(encodings, self.encoding)
        return min(distances), np.mean(distances), max(distances)

    def save_faces(self, base_dir):
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
    def load(cls, pathname, face_encodings):
        basename = os.path.basename(pathname)
        person = Person(basename)
        for face_filename in os.listdir(pathname):
            face_pathname = os.path.join(pathname, face_filename)
            image = cv2.imread(face_pathname)
            if image.size == 0:
                continue
            if face_filename in face_encodings:
                face_encoding = face_encodings[face_filename]
            else:
                print(pathname, face_filename, "calculate encoding")
                face_encoding = Face.get_encoding(image)
            if face_encoding is None:
                print(pathname, face_filename, "drop face")
            else:
                face = Face(face_filename, image, face_encoding)
                person.faces.append(face)
        print(person.name, "has", len(person.faces), "faces")
        person.calculate_average_encoding()
        return person

class PersonDB():
    def __init__(self):
        self.persons = []
        self.unknown_dir = "unknowns"
        self.encoding_file = "face_encodings"
        self.unknown = Person(self.unknown_dir)

    def load_db(self, dir_name):
        if not os.path.isdir(dir_name):
            return
        print("start loading persons in the directory", dir_name)
        start_time = time.time()

        # read face_encodings
        pathname = os.path.join(dir_name, self.encoding_file)
        try:
            with open(pathname, "rb") as f:
                face_encodings = pickle.load(f)
                print(len(face_encodings), "face_encodings in", pathname)
        except:
            face_encodings = {}

        # read persons
        for entry in os.scandir(dir_name):
            if entry.is_dir(follow_symlinks=False):
                pathname = os.path.join(dir_name, entry.name)
                person = Person.load(pathname, face_encodings)
                if len(person.faces) == 0:
                    continue
                if entry.name == self.unknown_dir:
                    self.unknown = person
                else:
                    self.persons.append(person)
        elapsed_time = time.time() - start_time
        print("loading persons finished in %.3f sec" % elapsed_time)

    def save_encodings(self, dir_name):
        face_encodings = {}
        for person in self.persons:
            for face in person.faces:
                face_encodings[face.filename] = face.encoding
        for face in self.unknown.faces:
            face_encodings[face.filename] = face.encoding
        pathname = os.path.join(dir_name, self.encoding_file)
        with open(pathname, "wb") as f:
            pickle.dump(face_encodings, f)
        print(pathname, "saved")

    def save_montages(self, dir_name):
        for person in self.persons:
            person.save_montages(dir_name)
        self.unknown.save_montages(dir_name)
        print("montages saved")

    def save_db(self, dir_name):
        print("start saving persons in the directory", dir_name)
        start_time = time.time()
        try:
            shutil.rmtree(dir_name)
        except OSError as e:
            pass
        os.mkdir(dir_name)

        for person in self.persons:
            person.save_faces(dir_name)
        self.unknown.save_faces(dir_name)

        self.save_montages(dir_name)
        self.save_encodings(dir_name)

        elapsed_time = time.time() - start_time
        print("saving persons finished in %.3f sec" % elapsed_time)

    def __repr__(self):
        s = "%d persons" % len(self.persons)
        num_known_faces = sum(len(person.faces) for person in self.persons)
        s += ", %d known faces" % num_known_faces
        s += ", %d unknown faces" % len(self.unknown.faces)
        return s

    def print_persons(self):
        print(self)
        persons = sorted(self.persons, key=lambda obj : obj.name)
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
    dir_name = "result"
    pdb = PersonDB()
    pdb.load_db(dir_name)
    pdb.print_persons()
    pdb.save_montages(dir_name)
    pdb.save_encodings(dir_name)
