#!/usr/bin/env python3

from person_db import Person
from person_db import Face
from person_db import PersonDB
import face_recognition
import numpy as np
from datetime import datetime
from datetime import timedelta
import cv2
import threading
import time
import traceback


class Observer():
    # called when Face Classifier is started
    def on_start(self, fc):
        pass

    # called when a person appears for the first time
    def on_new_person(self, person):
        pass

    # called when the person appears again
    def on_person(self, person):
        pass

    # called when Face Classifier is stopped
    def on_stop(self, fc):
        pass

class Observable():
    def __init__(self):
        self.observers = []

    def register_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    # notify that face classifier is started
    def notify_start(self):
        for observer in self.observers:
            observer.on_start(self)

    # notify that a person appears first
    def notify_new_person(self, person):
        for observer in self.observers:
            observer.on_new_person(person)

    # notify that the person appears again
    def notify_person(self, person):
        for observer in self.observers:
            observer.on_person(person)

    # notify that face classifier is stopped
    def notify_stop(self):
        for observer in self.observers:
            observer.on_stop(self)


class FaceClassifier(Observable):
    def __init__(self, person_db, settings):
        self.settings = settings
        self.last_frame = None
        self.running = False
        self.status_string = 'Face classifier is not running.'
        self.pdb = person_db
        Observable.__init__(self)

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

    # return list of dlib.rectangle
    def locate_faces(self, frame):
        #start_time = time.time()
        ratio = self.settings.resize_ratio
        if ratio == 1.0:
            rgb = frame[:, :, ::-1]
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)
            rgb = small_frame[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb)
        #elapsed_time = time.time() - start_time
        #print("locate_faces takes %.3f seconds" % elapsed_time)
        if ratio == 1.0:
            return boxes
        boxes_org_size = []
        for box in boxes:
            (top, right, bottom, left) = box
            left = int(left / ratio)
            right = int(right / ratio)
            top = int(top / ratio)
            bottom = int(bottom / ratio)
            box_org_size = (top, right, bottom, left)
            boxes_org_size.append(box_org_size)
        return boxes_org_size

    def detect_faces(self, frame):
        boxes = self.locate_faces(frame)
        if len(boxes) == 0:
            return []

        # faces found
        faces = []
        now = datetime.now()
        str_ms = now.strftime('%Y%m%d_%H%M%S.%f')[:-3] + '-'
        encodings = face_recognition.face_encodings(frame, boxes)
        for i, box in enumerate(boxes):
            face_image = self.get_face_image(frame, box)
            face = Face(str_ms + str(i) + ".png", face_image, encodings[i])
            face.location = box
            faces.append(face)
        return faces

    def compare_with_known_persons(self, face, persons):
        if len(persons) == 0:
            return None

        # see if the face is a match for the faces of known person
        encodings = [person.encoding for person in persons]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.settings.threshold:
            # face of known person
            persons[index].add_face(face)
            # re-calculate encoding
            persons[index].calculate_average_encoding()
            face.name = persons[index].name
            return persons[index]

    def compare_with_unknown_faces(self, face, unknown_faces):
        if len(unknown_faces) == 0:
            # this is the first face
            unknown_faces.append(face)
            face.name = "unknown"
            return

        encodings = [face.encoding for face in unknown_faces]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.settings.threshold:
            # two faces are similar - create new person with two faces
            person = Person()
            newly_known_face = unknown_faces.pop(index)
            person.add_face(newly_known_face)
            person.add_face(face)
            person.calculate_average_encoding()
            face.name = person.name
            newly_known_face.name = person.name
            return person
        else:
            # unknown face
            unknown_faces.append(face)
            face.name = "unknown"
            return None

    def draw_name(self, frame, face):
        color = (0, 0, 255)
        thickness = 2
        (top, right, bottom, left) = face.location

        # draw box
        width = 20
        if width > (right - left) // 3:
            width = (right - left) // 3
        height = 20
        if height > (bottom - top) // 3:
            height = (bottom - top) // 3
        cv2.line(frame, (left, top), (left+width, top), color, thickness)
        cv2.line(frame, (right, top), (right-width, top), color, thickness)
        cv2.line(frame, (left, bottom), (left+width, bottom), color, thickness)
        cv2.line(frame, (right, bottom), (right-width, bottom), color, thickness)
        cv2.line(frame, (left, top), (left, top+height), color, thickness)
        cv2.line(frame, (right, top), (right, top+height), color, thickness)
        cv2.line(frame, (left, bottom), (left, bottom-height), color, thickness)
        cv2.line(frame, (right, bottom), (right, bottom-height), color, thickness)

        # draw name
        #cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face.name, (left + 6, bottom + 30), font, 1.0,
                    (255, 255, 255), 1)

    def start_running(self):
        if self.running == True:
            print('already running')
            return

        src_file = self.settings.srcfile
        if src_file == '0':
            src_file = 0

        src = cv2.VideoCapture(src_file)
        if not src.isOpened():
            self.error_string = "cannot open inputfile"
            return -1

        frame_width = src.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = src.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_rate = src.get(5)
        self.frames_to_skip = int(round(self.frame_rate * self.settings.sbf))

        s = "* srcfile = " + self.settings.srcfile
        if self.settings.srcfile == '0':
            s += ' (webcam)'
        s += "\n* size = %dx%d" % (src.get(3), src.get(4))
        ratio = self.settings.resize_ratio
        s += "\n* resize_ratio = " + str(ratio)
        s += " -> %dx%d" % (int(src.get(3) * ratio), int(src.get(4) * ratio))
        s += "\n* frame_rate = %.3f f/s" % self.frame_rate
        s += "\n* process every " + str(self.frames_to_skip) + " frames"
        self.source_info_string = s

        self.src = src
        self.running = True
        t = threading.Thread(target=self.run)
        t.start()

    def stop_running(self):
        self.running = False
        self.status_string = 'Face classifier is not running.'

    def run(self):
        self.notify_start()
        frame_id = 0
        i = 0
        processing_time = 0
        while self.running:
            try:
                ret, frame = self.src.read()
                if frame is None:
                    break
                frame_id += 1
                if frame_id % self.frames_to_skip != 0:
                    continue

                start_time = time.time()
                self.process_frame(frame)
                processing_time += time.time() - start_time
                i += 1
                self.last_frame = frame

                dt = timedelta(seconds=int(frame_id/self.frame_rate))
                s = 'Face classifier running time: ' + str(dt) + '.'
                s += '\nTotal ' + str(i) + ' frames are processed.'
                s += '\nAverage processing time per frame is %.3f seconds.' % (processing_time / i)
                self.status_string = s
            except:
                trackback.print_exc()
                print('break loop...')
                break

        self.src.release()
        self.stop_running()
        self.pdb.save_db()
        self.pdb.print_persons()
        self.notify_stop()

    # this is core
    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        for face in faces:
            person = self.compare_with_known_persons(face, self.pdb.persons)
            if person:
                self.notify_person(person)
                person.update_last_face_time()
                continue
            person = self.compare_with_unknown_faces(face, self.pdb.unknown.faces)
            if person:
                self.pdb.persons.append(person)
                self.notify_new_person(person)
                person.update_last_face_time()

        # draw names
        for face in faces:
            self.draw_name(frame, face)

