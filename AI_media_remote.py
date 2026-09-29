import mediapipe as mp
import cv2
import pyautogui

mphands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 


hand = mphands.Hands(max_num_hands=1)

video = cv2.VideoCapture(0) 

while True:
    suc, img = video.read()
    if not suc or img is None:
        break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(img_rgb)
    
    tipids = [4, 8, 12, 16, 20]
    lmlst = []

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(landmarks.landmark):
                lmlst.append([id, lm.x, lm.y])
            
            if len(lmlst) >= 21:
                fingerlst = []

                # THUMB LOGIC
                if lmlst[4][1] > lmlst[3][1]:
                    fingerlst.append(1)
                else:
                    fingerlst.append(0)

                # 4 FINGERS LOGIC
                for i in range(1, 5):
                    if lmlst[tipids[i]][2] < lmlst[tipids[i]-2][2]: 
                        fingerlst.append(1)
                    else:
                        fingerlst.append(0)
                
                # Index Finger Up = Volume Up
                if fingerlst == [0, 1, 0, 0, 0]:
                    pyautogui.press("volumeup")
                    cv2.putText(img, "VOL UP", (50, 50), 1, 2, (0, 255, 0), 2)

                # Pinky Finger Up = Volume Down
                elif fingerlst == [0, 0, 0, 0, 1]:
                    pyautogui.press("volumedown")
                    cv2.putText(img, "VOL DOWN", (50, 50), 1, 2, (0, 0, 255), 2)

                # Peace Sign = Play/Pause
                elif fingerlst == [0, 1, 1, 0, 0]:
                    pyautogui.press("playpause")
                    cv2.putText(img, "PLAY/PAUSE", (50, 50), 1, 2, (255, 255, 0), 2)
                    cv2.waitKey(1000) 

                # Full Palm = Mute
                elif fingerlst == [1, 1, 1, 1, 1]:
                    pyautogui.press("volumemute")
                    cv2.putText(img, "MUTE", (50, 50), 1, 2, (255, 0, 0), 2)
                    cv2.waitKey(1000)

            mp_drawing.draw_landmarks(img, landmarks, mphands.HAND_CONNECTIONS)

    cv2.imshow("Simplified AI Remote", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()