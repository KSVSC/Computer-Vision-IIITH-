import matplotlib.pyplot as plt
import numpy as np
import itertools
import warnings
import math
import cv2
import os
warnings.filterwarnings("ignore")

def ORB_feature_matching(images, num_matches):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb = cv2.ORB_create()
    list1 = []
    list2 = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(img_gray, None)
        list1.append(kp)
        list2.append(des)

    matches = sorted(bf.match(list2[0], list2[1]), key=lambda x: x.distance)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=None,
                       flags=2)
    im = cv2.drawMatches(images[0], list1[0], images[1],
                         list1[1], matches[:num_matches], None, **draw_params)
    return matches, list1, list2, im

def SIFT_feature_matching(images, num_matches):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    orb = cv2.SIFT_create()
    list1 = []
    list2 = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(img_gray, None)
        list1.append(kp)
        list2.append(des)

    matches = sorted(bf.match(list2[0], list2[1]), key=lambda x: x.distance)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=None,
                       flags=2)
    im = cv2.drawMatches(images[0], list1[0], images[1],
                         list1[1], matches[:num_matches], None, **draw_params)
    return matches, list1, list2, im


def SIFT(im, step_size):
    h, w = im.shape[:2]
    kp = []
    # sift = cv2.SIFT_create()
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(0, h, step_size):
        for j in range(0, w, step_size):
            kp.append(cv2.KeyPoint(float(i), float(j), float(step_size)))
    return sift.compute(im, kp)


def extract_features(image_list):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    kp = []
    des = []
    im_new_list = []
    for file in image_list:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2. COLOR_BGR2RGB)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = None
        faces = face_cascade.detectMultiScale(image, 1.1, 4)
        if(len(faces) != 0):
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                faces = img[y:y + h, x:x + w]
            if faces is not None and len(faces) > 0:
                faces = cv2.resize(faces, (250, 250))
                im_new_list.append(faces)
                sift1,sift2 = SIFT(np.array(faces), 10)
                kp.append(sift1)
                des.append(sift2)

    matches = bf.match(des[0], des[1])
    draw_params = dict(matchColor=(0,250,0),singlePointColor=None,matchesMask=None,flags=2)
    im = cv2.drawMatches(im_new_list[0], kp[0], im_new_list[1], kp[1], matches[:200], None, **draw_params)
    return matches, kp, des, im

img_list = []
for i in list([7, 6]):
    im_path = "../dataset/Ana_Guevara/Ana_Guevara_000"+str(i)+".jpg"
    img_list.append(im_path)
matches, list1, list2, im = extract_features(img_list)
fig = plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(im)
plt.show()