import cv2
import os
import numpy

haar_file = "C:/Users/proho/Desktop/Jetlearn/OpenCV/Leso9/haarcascade_frontalface_default.xml"
datasets = "C:/Users/proho/Desktop/Jetlearn/OpenCV/Leso9/datasets"

(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        print(names)
        subjectpath = os.path.join(datasets, subdir)

        for filemame in os.listdir(subjectpath):
            path = subjectpath + '/' + filemame
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
(width, height) = (130, 100)

recogniser = cv2.LBPHFaceRecognizer_create()
recogniser.train(images, labels)
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale( gray, scaleFactor, minNeighbours)
    faces = face_cascade.detectMultiScale(gray,1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(im, (x,y), (x + w, y + h), (255,0,0), 2)
        face = gray [y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        #try recognise the face
        prediction = recogniser.predict(face_resize)
        print(prediction)
        cv2.rectangle(im, (x,y), (x + w, y + h), (0,255,0), 3)

        if prediction[1]<500:
           cv2.putText(im, '% s - %.0f' %(names[prediction[0]], prediction[1]), (x-10, y-10),
cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
          cv2.putText(im, 'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.imshow('OpenCV', im)

    key = cv2.waitKey(10)
    #space key
    if key == 27:
        break