import cv2
import mediapipe as mp

webcam=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
my_hands=mp_hands.Hands()
utils=mp.solutions.drawing_utils
drawing_spec = utils.DrawingSpec((0,0,255))
drawing_spec1 = utils.DrawingSpec((0,255,0))
while True:
    ret,frame=webcam.read()  
    frame=cv2.flip(frame,1)
    if not ret:
        break
    rgb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output=my_hands.process(rgb_image)
    hands=output.multi_hand_landmarks

    if hands:
        for hand in hands:
            utils.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS,drawing_spec,drawing_spec1)
    
    cv2.imshow("HAND LANDMARK",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()