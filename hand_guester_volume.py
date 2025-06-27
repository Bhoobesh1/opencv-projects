import cv2
import mediapipe as mp
import pyautogui

webcam=cv2.VideoCapture(0)
my_hands=mp.solutions.hands.Hands() 
utils=mp.solutions.utils 

while True:
    ret,frame=webcam.read() 
    if not ret:
        break
    rgb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output=my_hands.process(rgb_image)
    hands=output.multi_hand_landmarks

    cv2.imshow("hand volume control using python",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()