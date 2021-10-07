import cv2

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("Cascade/haarcascade_eye.xml")


while True:
    _,img = video.read()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 2, minNeighbors = 5)
    eyes = eye_cascade.detectMultiScale(gray_img, scaleFactor = 2, minNeighbors = 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        for (x1,y1,w1,h1) in eyes:
            cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
    cv2.imshow('img',img)
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()
                
        
 
