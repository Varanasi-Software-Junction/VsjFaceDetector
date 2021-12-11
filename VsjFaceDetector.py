import cv2
import sys
import logging as log
import datetime as dt
from time import sleep


# https://pypi.org/project/opencv-python/
def webCamDetector():


    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = getClassifier()
    log.basicConfig(filename='webcam.log', level=log.INFO)

    video_capture = cv2.VideoCapture(0)
    anterior = 0

    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def videoDetector(videopath):
    face_cascade = getClassifier()
    cap = cv2.VideoCapture(videopath)
    while True:
        x, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        highlightFaces(faces,img)
        cv2.imshow('Video', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
def getDetectedFaces(imagepath):

    # https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    cascadeclassifier = getClassifier()
    image = cv2.imread(imagepath)
    # https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
    grayscaleimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    faces = cascadeclassifier.detectMultiScale(grayscaleimage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), )
    # https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html
    return image, faces


def highlightFaces(faces, image):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html

def getClassifier():
    cascadeclassifierpath = "cascadeclassifier.xml"
    # https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    return cv2.CascadeClassifier(cascadeclassifierpath)