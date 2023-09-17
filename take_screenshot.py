import numpy as np
import pyautogui
import imutils
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)


    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Accessing the landmarks by their position
            lm_list=[]
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Array to hold true or false if finger is folded    
            finger_fold_status =[]
            for tip in finger_tips:
                # Getting the landmark tip position and drawing green circle
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)

                # Writing condition to check if finger is folded i.e checking if finger tip starting value is smaller than finger starting position which is inner landmark. for index finger    
                # If finger folded changing color to blue
                if lm_list[tip].x > lm_list[tip - 3].x:
                    cv2.circle(img, (x,y), 15, (255, 0, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

             # Checking if all fingers are folded
            if all(finger_fold_status):
                image = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cv2.imwrite("in_memory_to_disk.png", image)
                pyautogui.screenshot("straight_to_disk.png")
                image = cv2.imread("straight_to_disk.png")
                cv2.imshow("screenshot", imutils.resize(image, width = 600))

            mp_draw.draw_landmarks(img, hand_landmark,
            mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
            mp_draw.DrawingSpec((0,255,0),4,2))
    

    cv2.imshow("hand tracking", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()