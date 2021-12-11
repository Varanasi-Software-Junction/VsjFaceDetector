import VsjFaceDetector as vsj
import cv2
vsj.webCamDetector()
'''
videopath="test.mp4"
vsj.videoDetector(videopath)
imagePath = "facebook2.jpg"
image, faces = vsj.getDetectedFaces(imagePath)
vsj.highlightFaces(faces, image)
n = len(faces)
cv2.imshow("Varanasi Software Junction --- " + str(n) + " faces found", image)
# https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
cv2.waitKey(0)
# https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
'''