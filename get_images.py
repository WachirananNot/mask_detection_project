import os
import cv2
import tensorflow
import keras
from PIL import Image

webcam = cv2.VideoCapture(0)

face_cascade = 'haarcascade_frontalface_default.xml'
count = 0
x = 0
while x != 1 and x != 2:
    x = int(input("Select mode to get image:\n1 = non_mask\n2 = mask\n"))
    if x == 1:
        folder_name = f"non_mask"
    else:
        folder_name = f"mask"
if not os.path.exists("train/"+folder_name):
    os.makedirs("train/"+folder_name)
while True:

    success, image_bgr = webcam.read()
    image_org =  image_bgr.copy()
    image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(face_cascade)
    faces = face_classifier.detectMultiScale(image_bw)

    print(f'There are {len(faces)} faces found.')

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image_file = os.path.join("train/"+folder_name, f"{folder_name}{count}.jpg")
        cv2.imwrite(image_file,image_org[y:y+h,x:x+w])
        count += 1
    cv2.imshow("Faces found", image_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
