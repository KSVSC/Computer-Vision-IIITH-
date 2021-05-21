import numpy as np
import os
import cv2
import glob
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from lmnn import LMNN
from mknn import MKNN
from sklearn.neighbors import KNeighborsClassifier
file_path = []


def SIFT(im,step_size):
    h,w=im.shape[:2]
    kp=[]
    # sift = cv2.SIFT_create()
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(0,h,step_size):
        for j in range(0,w,step_size):
            kp.append(cv2.KeyPoint(float(i),float(j),float(step_size)))
          
    keypoints,descriptor=sift.compute(im,kp)
      
    return descriptor.reshape(-1)


def extract_features(image_list,Y):
    features = []
    Y_new = []
    i = -1
    for file in image_list:
        i = i + 1
        img = cv2.imread(file)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = None
        faces = face_cascade.detectMultiScale(image, 1.1, 4)
        if(len(faces)!=0):
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                faces = img[y:y + h, x:x + w]
            if faces is not None and len(faces)>0:
                faces = cv2.resize(faces, (250,250))
                sift = SIFT(np.array(faces),10)
                features.append(sift)
                Y_new.append(Y[i])
    features = np.array(features)
    return features,np.array(Y_new)

def preprocess(dir):
    '''
        This method preprocesses the dataset and returns two array X and Y
        X: contains the original data
        Y: contains the labels
        input:
            dir -> Directory of the dataset.
    '''
    dirlist = os.listdir(dir)
    X = []
    Y = []
    for path in tqdm(dirlist):
        file = glob.glob(dir + '/' + path + '/*.jpg')
        for i in file:
            file_path.append(i)
            img = cv2.imread(i)
            img = img.reshape((250, 250, 3))
            X.append(img.reshape(-1))
            Y.append(path)
    X = np.array(X)
    Y = np.array(Y)
    uniquey = enumerate(np.unique(Y))
    uniquey = {val: i for i, val in uniquey}
    y = []
    for i in Y:
        y.append(uniquey[i])
    Y = np.array(y)
    return X, Y, file_path


def select_labels(X_train, Y_train, k=3):
    '''
        This method discards lables with less than k no of samples
        input:
            X_train -> data
            Y_train -> labels
            k -> min no of samples required per label(default 3)
        returns:
            X, Y
    '''
    classes = []
    for i in np.unique(Y_train):
        if np.sum(Y_train == i) < k:
            classes.append(i)

    classes = np.array(classes)
    X_new = []
    Y_new = []
    for i in range(X_train.shape[0]):
        if Y_train[i] not in classes:
            X_new.append(X_train[i])
            Y_new.append(Y_train[i])
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new

if __name__ == '__main__':
    dir = '../dataset'

    print('Data Preprocessing...')
    X, Y, file_path = preprocess(dir)
    # X = X[:1000]
    # Y = Y[:1000]
    print('Data preprocessing done...')
    print('Data Shape:', X.shape)
    print('label Shape:', Y.shape)
 

    features,Y = extract_features(file_path,Y)
    X = features

    # Normalize the dataset
    mean = np.mean(X)
    var = np.var(X)
    X = (X - mean)/var
    print('data normalized')

    # Features contain both keypoints and descriptors of the images.
    # Reduce dimensionality if needed
    pca = PCA(n_components=200)
    X_pca = pca.fit(X).transform(X)
    print('dimension reduced..')
    print('data shape:', X_pca.shape)
    print('label shape:', Y.shape)

    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_pca, Y, test_size=0.2, random_state=42)
    print(X_train.shape)

    X_new, Y_new = select_labels(X_train, Y_train, 5)
    print(X_new.shape, Y_new.shape)
    X_new_test, Y_new_test = select_labels(X_test, Y_test, 5)
    print('Performing KNN')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_new, Y_new)
    pred1 = neigh.predict(X_new_test)
    acc1 = np.sum(pred1 == Y_new_test)/X_new_test.shape[0]
    print('Accuracy before LMNN: ', acc1*100)

    # LMNN Transform of data from Euclidean space.
    lmnn = LMNN()
    lmnn = lmnn.fit(X_new, Y_new)
    X_transformed = lmnn.transform(X_new)
    X_test_transformed = lmnn.transform(X_new_test)

    print('Performing KNN on LMNN')
    neigh1 = KNeighborsClassifier(n_neighbors=3)
    neigh1.fit(X_transformed, Y_new)
    pred1 = neigh1.predict(X_test_transformed)
    acc2 = np.sum(pred1 == Y_new_test)/X_new_test.shape[0]
    print('Accuracy after LMNN+KNN: ', acc2*100)

    print('Performing MKNN')
    mknn = MKNN(X_new,Y_new)
    predicted_label, predicted_probab = mknn.marginalized_knn(X_new_test, k=5)
    #pred2 = neigh.predict(X_test_transformed)
    acc2 = np.sum(predicted_label == Y_new_test)/X_new_test.shape[0]    
    print('Accuracy after MKNN: ', acc2*100)

    print('Performing MKNN on LMNN')
    mknn = MKNN(X_transformed,Y_new)
    predicted_label, predicted_probab = mknn.marginalized_knn(X_test_transformed, k=5)
    #pred2 = neigh.predict(X_test_transformed)
    acc2 = np.sum(predicted_label == Y_new_test)/X_new_test.shape[0]    
    print('Accuracy after LMNN+MKNN: ', acc2*100)
