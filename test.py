import numpy as np
import cv2
import pickle
import datetime

#Capture images
class FaceDetector():
    """
       A class that detect faces and indentify the specific person
       display camera position
    """

    def __init__(self,video_source, trainning_data, labels, camera_number):
        """Initialize all the atribute in this class"""

        # Initialize and Read the video file
        self.video = cv2.VideoCapture(video_source)
        # Initialize and Create cascade classifier
        self.face_cascade = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")
        #Initialize and Create Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Read the trinnig faces yml data
        self.recognizer.read(trainning_data)
        self.camera_number = camera_number

        # Initialize faces name and index number
        self.labels = {}
        # Open face name and index number and insert into
        with open(labels,"rb") as f:
            og_labels = pickle.load(f)
            self.labels = {v:k for k,v in og_labels.items()}


    def draw_text(self,img,text,position,color):
        """Draw Text into the images"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        stroke = 2
        cv2.putText(img,text,position,font,1,color,stroke,cv2.LINE_AA)

    def detect_faces(self):
        """This method detect faces and create rectangle"""
        while True:
            _, img = self.video.read()
            #cover into gray image
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            #detect faces
            faces = self.face_cascade.detectMultiScale(gray_img, scaleFactor = 1.35, minNeighbors = 5)
            #draw rectangle on the face
            for (x,y,w,h) in faces:
                roi_gray = gray_img[y:y+h, x:x+w]
                 # recognize using deep learned modules
                id_, conf = self.recognizer.predict(roi_gray)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                if conf >= 50 and conf <=80:
                    name = self.labels[id_]
                    time = str(datetime.datetime.now())
                    date = time[:11]
                    time = time[11:]
                    self.draw_text(img,name,(x,y),(255,255,255))
                    self.draw_text(img,'camera_no:' + str(self.camera_number),(20,20),(255,0,0))
                    self.draw_text(img,'Date:' + date,(20,60),(255,0,0))
                    self.draw_text(img,'Time:' + time,(20,100),(255,0,0))
                cv2.imwrite("Me.png",roi_gray)
            #display the image
            cv2.imshow("Face Detector",img)
            cv2.waitKey(1)

        self.video.release()
        cv2.destroyAllWindows()

camera_1 = FaceDetector(0,'trainner.yml','labels.pickle',1)
camera_1.detect_faces()

camera_2 = FaceDetector(0,'trainner.yml','labels.pickle',2)
camera_2.detect_faces()
