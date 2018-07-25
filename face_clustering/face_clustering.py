#!/usr/bin/env python3
import face_recognition
import cv2

class Face():
    def __init__(self, frame_id, name, box, encoding):
        self.frame_id = frame_id
        self.name = name
        self.box = box
        self.encoding = encoding

class FaceClustering():
    def __init__(self, src):
        print("%dx%d, %d frame/sec" % (src.get(3), src.get(4), src.get(5)))
        self.faces = []
        self.src = src
        self.num_frame = 0
        self.frame_id = 0
        self.frame_rate = round(src.get(5))

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

        self.num_frame += 1
        if self.num_frame % self.frame_rate != 0:
            return True

        frame_id = self.num_frame // self.frame_rate
        print("frame_id =", frame_id)

        rgb = frame[:, :, ::-1]
        faces_in_frame = self.quantify(frame_id, rgb)

        #if (len(faces_in_frame) == 0):
        #    return True

        print(faces_in_frame)

        # show the frame
        self.drawBox(frame, faces_in_frame)
        fname = "frame_" + str(frame_id) + ".jpg"
        cv2.imwrite(fname, frame)
        #cv2.imshow("Frame", frame)

        self.faces.extend(faces_in_frame)

        return True


if __name__ == '__main__':
    #cam = cv2.VideoCapture(0)
    cam = cv2.VideoCapture('LaLaLand-SummerMontage_Madeline.mp4')
    fc = FaceClustering(cam)

    while True:
        ret = fc.run()
        if ret is False:
            break

    # do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()
    print('finish')
