import numpy as np
import cv2
import pickle

first_frame=None
status_list=[None,None]
#Capture images
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle","rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


while True:
    _, frame = video.read()
    h,w,_ = frame.shape
    img = cv2.resize(frame,(w,h))
    #cover into gray image
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img,(21,21),0)

    
    if first_frame is None:
        first_frame=gray_img
        continue


    delta_frame=cv2.absdiff(first_frame,gray_img)
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    (_,cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    #detect faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.35, minNeighbors = 5)
    #draw rectangle on the face
    for (x,y,w,h) in faces:
        roi_gray = gray_img[y:y+h, x:x+w]
         # recognize using deep learned modules
        id_, conf = recognizer.predict(roi_gray)
        cv2.rectangle(gray_img,(x,y),(x+w,y+h),(0,0,255),2)
        if conf >= 50 and conf <=80:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        cv2.imwrite("Me.png",roi_gray)
    #display the image
    cv2.imshow("Frame",img)
    cv2.imshow("Frame",gray_img)
    cv2.imshow("Frame",thresh_frame)
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()
