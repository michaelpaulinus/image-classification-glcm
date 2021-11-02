import matplotlib.pyplot as plt
import matplotlib

from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread
from skimage import color, data

import numpy as np
import pandas as pd
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB # import Naive Bayes classifier
from sklearn.datasets import load_iris
from sklearn import svm

from main import X_test, X_train # import Support Vector Machines classifier

# store image folder into variable
path_normal = glob.glob("C:\\Users\\Student\\Desktop\\Varsity\\ENEL4AI Artificial Intelligence\\Project\\AI\\chest_xray_dataset\\train\\NORMAL\\*.jpeg")
path_pneumonia = glob.glob("C:\\Users\\Student\\Desktop\\Varsity\\ENEL4AI Artificial Intelligence\\Project\\AI\\chest_xray_dataset\\train\\PNEUMONIA\\*.jpeg")

path_normaltest = glob.glob("C:\\Users\\Student\\Desktop\\Varsity\\ENEL4AI Artificial Intelligence\\Project\\AI\\chest_xray_dataset\\test\\NORMAL\\*.jpeg")
path_pneumoniatest = glob.glob("C:\\Users\\Student\\Desktop\\Varsity\\ENEL4AI Artificial Intelligence\\Project\\AI\\chest_xray_dataset\\test\\PNEUMONIA\\*.jpeg")

#array to store image files
normal_img = []
pneumonia_img = []

normal_imgtest = []
pneumonia_imgtest = []

#loop to read images into an array
for img in path_normal:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    normal_img.append(n)

for img in path_pneumonia:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    pneumonia_img.append(n)

for img in path_normaltest:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    normal_imgtest.append(n)

for img in path_pneumoniatest:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    pneumonia_imgtest.append(n)

img_props=[]
X = []
y = []

img_propst = []
Xt = []
yt = []

# extract haralick features of normal chest
for img in normal_img:
    # create glcm of an image (distance=[1,2], angles=[0,45,90,135])
    g = greycomatrix(img, [1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    contrast = greycoprops(g, 'contrast')
    energy = greycoprops(g, 'energy')
    homogeneity = greycoprops(g, 'homogeneity')
    dissimilarity = greycoprops(g, 'dissimilarity')
    img_class = 0 # normal

    X.append(np.array([contrast, energy, homogeneity, dissimilarity]))
    y.append(img_class)

    props = np.array([contrast, energy, homogeneity, dissimilarity, img_class])
    img_props.append(props)


# extract haralick features of pneumonia chest
for img in pneumonia_img:
    # create glcm of an image (distance=[1,2], angles=[0,45,90,135])
    g = greycomatrix(img,[1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    contrast = greycoprops(g, 'contrast')
    energy = greycoprops(g, 'energy')
    homogeneity = greycoprops(g, 'homogeneity')
    dissimilarity = greycoprops(g, 'dissimilarity')
    img_class = 1 # pneumonia

    X.append(np.array([contrast, energy, homogeneity, dissimilarity]))
    y.append(img_class)

    props = np.array([contrast, energy, homogeneity, dissimilarity, img_class])
    img_props.append(props)

X = np.array(X).reshape((89,32)) # 5276 = number of images trained, 32 = number of features

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=2/10) # split train=0.8, test=0.2
X_train = X
y_train = y

############################ TEST FOLDER ############################
# extract haralick features of normal chest
for img in normal_imgtest:
    # create glcm of an image (distance=[1,2], angles=[0,45,90,135])
    g = greycomatrix(img, [1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    contrast = greycoprops(g, 'contrast')
    energy = greycoprops(g, 'energy')
    homogeneity = greycoprops(g, 'homogeneity')
    dissimilarity = greycoprops(g, 'dissimilarity')
    img_class = 0 # normal

    Xt.append(np.array([contrast, energy, homogeneity, dissimilarity]))
    yt.append(img_class)

    props = np.array([contrast, energy, homogeneity, dissimilarity, img_class])
    img_propst.append(props)


# extract haralick features of pneumonia chest
for img in pneumonia_imgtest:
    # create glcm of an image (distance=[1,2], angles=[0,45,90,135])
    g = greycomatrix(img,[1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    contrast = greycoprops(g, 'contrast')
    energy = greycoprops(g, 'energy')
    homogeneity = greycoprops(g, 'homogeneity')
    dissimilarity = greycoprops(g, 'dissimilarity')
    img_class = 1 # pneumonia

    Xt.append(np.array([contrast, energy, homogeneity, dissimilarity]))
    yt.append(img_class)

    props = np.array([contrast, energy, homogeneity, dissimilarity, img_class])
    img_propst.append(props)

Xt = np.array(Xt).reshape((89,32)) # 5276 = number of images trained, 32 = number of features
X_test = Xt
y_test = yt

####### MLP CLASSIFIER #######
# generating the model
clf_mlp = MLPClassifier(solver='adam', alpha=0.0001,hidden_layer_sizes=(20, 30),max_iter=600 )
# train the model
clf_mlp.fit(X_train, y_train)
# results
print('############################ RESULTS ############################')
# predict the response
predictions_mlp = clf_mlp.predict(X_test)
# accuracy score
print('- Accuracy: ', metrics.accuracy_score(y_test, predictions_mlp))
# precision score
print('- Precision: ', metrics.recall_score(y_test, predictions_mlp))
# summary
print("- Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != predictions_mlp).sum()))
print('Test: ', y_test)
print('Predicted: ', predictions_mlp)
print('############################ END OF RESULTS ############################')


####### NBC CLASSIFIER #######
# generating the model
gnb = GaussianNB()
# train the model
gnb.fit(X_train, y_train)
# results
print('############################ RESULTS ############################')
# predict the response
predictions_gnb = gnb.predict(X_test)
# accuracy score
print('- Accuracy: ', metrics.accuracy_score(y_test, predictions_gnb))
# precision score
print('- Precision: ', metrics.recall_score(y_test, predictions_gnb))
# summary
print("- Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != predictions_gnb).sum()))
print('- Test: ', y_test)
print('- Predicted: ', predictions_gnb)
print('############################ END OF RESULTS ############################')

####### SVM CLASSIFIER #######
# generating the model
clf_svm = svm.SVC()
# train the model
clf_svm.fit(X_train, y_train)
# results
print('############################ RESULTS ############################')
# predict the response
predictions_svm = clf_svm.predict(X_test)
# accuracy score
print('- Accuracy: ', metrics.accuracy_score(y_test, predictions_svm))
# precision score
print('- Precision: ', metrics.recall_score(y_test, predictions_svm))
# summary
print("- Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != predictions_svm).sum()))
print('- Test: ', y_test)
print('- Predicted: ', predictions_svm)
print('############################ END OF RESULTS ############################')