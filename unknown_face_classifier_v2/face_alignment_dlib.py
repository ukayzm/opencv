# this code is taken from https://github.com/nlhkh/face-alignment-dlib
# and adjusted

import numpy as np
import cv2
import dlib

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def rotate_face(predictor, img, rect):
    # dlib.rectangle = [(left, top), (right, bottom)]
    height, width = img.shape[:2]
    shape = predictor(img, rect)
    left_eye = extract_left_eye_center(shape)
    right_eye = extract_right_eye_center(shape)

    M = get_rotation_matrix(left_eye, right_eye)
    rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)
    return rotated

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]

def get_aligned_face(predictor, face_image):
    height, width = face_image.shape[:2]
    x = int(width / 3)
    y = int(height / 3)
    rect_of_face = dlib.rectangle(x, y, x*2, y*2)

    # rotate the face image
    rotated_image = rotate_face(predictor, face_image, rect_of_face)

    # resize the image
    resized_image = cv2.resize(rotated_image, dsize=(128*3, 128*3),
                               interpolation=cv2.INTER_LINEAR)
    return resized_image

