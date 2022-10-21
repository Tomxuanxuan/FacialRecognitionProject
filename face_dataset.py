# !/usr/bin/env python
# encoding: utf-8
"""
@author: tx
@file: face_dataset.py
@time: 2022/9/22 3:29 PM
@desc: 人脸数据库 存储人脸数据
"""
import cv2
import os
from cv2.data import haarcascades

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

base_path = os.path.dirname(os.path.abspath(__file__))
print("base_path", base_path)
print("haarcascades", haarcascades)

face_detector = cv2.CascadeClassifier('{0}/{1}'.format(haarcascades, 'haarcascade_frontalface_default.xml'))

face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while (True):
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        print("count", count)

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face sample and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()