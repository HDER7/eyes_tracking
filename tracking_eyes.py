import cv2
import dlib
import numpy as np


def shape_numpy(shape, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)
    return coordinates


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, (0, 255, 0))
    return mask


def count(tresh, mid, img, right=False):
    cs, _ = cv2.findContours(tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        c = max(cs, key= cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx,cy), 4 , (0, 255, 0), 2)
    except:
        pass


detector = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor('shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
tresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)


def nothing(x):
    pass


cv2.createTrackbar('treshold', 'image', 0, 255, nothing)

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predict(gray, rect)
        shape = shape_numpy(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel,5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0].all(axis=2))
        eyes[mask] = [255, 255, 255]

