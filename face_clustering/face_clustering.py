#!/usr/bin/env python3
import face_recognition
import cv2

class Face():
    def __init__(self, frame_no, name, box, encoding):
        self.frame_no = frame_no
        self.name = name
        self.box = box
        self.encoding = encoding

class FaceClustering():
    def quantify(self, frameId, rgb):
        faces_in_frame = []
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        for box, encoding in zip(boxes, encodings):
            face = Face(frameId, None, box, encoding)
            faces_in_frame.append(face)
        return faces_in_frame

    def drawBox(self, frame, faces_in_frame):
        # Draw a box around the face
        for face in faces_in_frame:
            (top, right, bottom, left) = face.box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


if __name__ == '__main__':
    import math
    import datetime

    cam = cv2.VideoCapture(0)
    frameRate = cam.get(5)
    print("frameRate =", frameRate)
    frame_no = 0
    prevFrameId = frame_no // math.floor(frameRate)
    fc = FaceClustering()

    faces = []

    while (cam.isOpened()):
        ret, frame = cam.read()
        frame_no += 1
        frameId = frame_no // math.floor(frameRate)
        if (frameId == prevFrameId):
            continue
        prevFrameId = frameId
        print("frameId =", frameId)

        rgb = frame[:, :, ::-1]
        faces_in_frame = fc.quantify(frameId, rgb)

        #if (len(faces_in_frame) == 0):
        #    continue

        print(faces_in_frame)

        # show the frame
        fc.drawBox(frame, faces_in_frame)
        cv2.imshow("Frame", frame)

        faces.extend(faces_in_frame)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()
    print('finish')
