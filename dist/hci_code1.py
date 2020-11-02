
from scipy.spatial import distance as dist

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pyautogui
import math

def midPoint(point1, point2):
    midPoint = (((point1[0] + point2[0]) / 2), ((point1[1] + point2[1]) / 2))
    return midPoint


def distance(point1, point2):
    distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    return distance

def click():
    pom = pyautogui.position()
    pyautogui.click(pom)

def nothing(x):
    pass

def moveCursor(direction):
    if direction=="right":
        pyautogui.moveRel(40,0,0)
    elif direction=="left":
        pyautogui.moveRel(-40,0,0)
    elif direction=="up":
        pyautogui.moveRel(0,80,0)
    elif direction=="down":
        pyautogui.moveRel(0,-80,0)


def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])


    C = dist.euclidean(eye[0], eye[3])


    ear = (A + B) / (2.0 * C)


    return ear


cv2.namedWindow('HCI',cv2.WINDOW_NORMAL)



EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3


COUNTER = 0
TOTAL = 0



detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


cap=cv2.VideoCapture(0)

fileStream = False

lx=0
ly=0
rx=0
ry=0
ol = (0,0)

while True:

    ret,frame=cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    rects = detector(gray, 0)


    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)


        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        thresh = 4
        ear = (leftEAR + rightEAR) / 2.0
        diff = abs(leftEAR-rightEAR)
        diff = diff*100
        print(diff, thresh)
        if leftEAR>rightEAR:
            if diff>thresh:
                moveCursor("left")
        if rightEAR>leftEAR:
            if diff > thresh:
                moveCursor("right")

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        LM = cv2.moments(leftEyeHull)
        RM = cv2.moments(rightEyeHull)
        if LM['m00']>0:
            lx=int((LM["m10"]/LM["m00"]+ 1e-7)*1)
            ly = int((LM["m01"] / LM["m00"] + 1e-7) * 1)
        if RM['m00']>0:
            rx=int((RM["m10"]/RM["m00"]+ 1e-7)*1)
            ry = int((RM["m01"] / RM["m00"] + 1e-7)*1)

        mid = midPoint((lx,ly),(rx,ry))
        d = distance(mid,ol)
        print(d)
        if d>4:
            if mid[1]<ol[1]:
                moveCursor("down")
            else:
                moveCursor("up")
        ol=mid
        if ear < EYE_AR_THRESH:
            COUNTER += 1


        else:

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                #click()


            COUNTER = 0


        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "lEAR: {:.2f}".format(leftEAR), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "rEAR: {:.2f}".format(rightEAR), (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()
