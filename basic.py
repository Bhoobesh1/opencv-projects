import cv2
cap=cv2.VideoCapture(0) # to open the webcam

while True:
   ret,frame=cap.read() #read one frame
   if not ret:
      break
   cv2.imshow("webcam",frame) #show the webcam with title webcam
   if cv2.waitKey(1)& 0XFF ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
#cv2.waitkey(1) is waits 1 milliseconds