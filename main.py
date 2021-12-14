import cv2, imutils, os, math
from tkinter import *
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import Functions

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
Xp, Yp = 0,0
close = 0
width = 5
color = (0, 0, 255)
folderPath = "background"
myList = os.listdir(folderPath)
overlayList=[]
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[1]
dock = overlayList[4]
cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), np.uint8)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image =cv2.blur(image, (3,3))
        image = cv2.flip(image, 1)

        image[0:80, 0:640] = header
        image[220:420, 560:640] = dock


        Alto, Ancho, _ = image.shape
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2))

                X2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * Ancho)
                Y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * Alto)
                X1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * Ancho)
                Y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * Alto)
                X3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * Ancho)
                Y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * Alto)
                X4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * Ancho)
                Y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * Alto)

                State= Functions.distances(X1, Y1, X2, Y2)
                CloseEstate = Functions.distances(X3, Y3, X4, Y4)
                cv2.circle(image, (X1, Y1), 3, color, cv2.FILLED) # (182, 149, 192)
                fingerStat, count = Functions.countFingers(image, results)
                gesture = Functions.recognizeGestures(image, fingerStat, count, False, False)
                keys = list(fingerStat)
                key = list(gesture.values())
                if count['LEFT'] == 0:
                    hand = 'RIGHT'
                    Handgesture = key[0]
                    keys = keys[0:5]
                elif count['RIGHT'] ==0:
                    hand = 'LEFT'
                    Handgesture = key[1]
                    keys = keys[5:10]

                print(Handgesture)
                indice = keys[1]
                medio = keys[2]

                #Selecting mode
                if (fingerStat[indice] and fingerStat[medio]) and State :
                    cv2.circle(image, (X2, Y2), 5, (182, 149, 192), 2)
                    cv2.putText(image, "Selecting", (500, 450), 3, 0.5, (0,0,255), 1)
                    if Y2 <= 85:
                        width, header, color = Functions.Selection(X2, Y2, overlayList, color, width, header)
                    if  540 < X2 <640:
                        if 240< Y2 < 275:
                            dock = overlayList[5]
                            canvas = np.zeros((480, 640, 3), np.uint8)
                        elif 326 < Y2 < 370:
                            dock = overlayList[6]
                            Functions.Save(canvas)
                            cv2.putText(image, "Saved", (200, 200), 3, 1, (255, 0, 0), 2)
                    else: dock = overlayList[4]

                elif Handgesture == 'V SIGN':
                    dock = overlayList[5]
                    canvas = np.zeros((480, 640, 3), np.uint8)

                elif Handgesture == 'SPIDERMAN SIGN':
                    dock = overlayList[6]
                    Functions.Save(canvas)
                    cv2.putText(image, "Saved", (200, 200), 3, 1, (255, 0, 0), 2)
                #Drawing mode
                elif (fingerStat[indice] == True ) and (count[hand] == 1 ):
                    if Xp == 0 and Yp ==0:
                        Xp, Yp = X1, Y1

                    cv2.line(canvas, (Xp, Yp), (X1, Y1), color, width)
                    cv2.putText(image, "Drawing", (500, 450), 3, 0.5, (0,0,255), 1)
                    #cv2.circle(image, (X1, Y1), 3, (0, 200, 0), cv2.FILLED)
                    Xp = X1
                    Yp = Y1

                #Closing
                elif CloseEstate and count[hand] ==0 :
                    close = CloseEstate
                    cv2.putText(image, "CERRANDO...", (140, 180), 3, 2, (0 ,0 ,255), 3)
                else:
                    Xp, Yp = 0, 0
                    dock = overlayList[4]

        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_and(image, imgInv)
        image = cv2.bitwise_or(image, canvas)
        
        cv2.imshow('MediaPipe Hands', image)
        cv2.imshow('Canvas', canvas)
        if cv2.waitKey(5) & 0xFF == 32 or close:
            break
cap.release()
cv2.destroyAllWindows()



