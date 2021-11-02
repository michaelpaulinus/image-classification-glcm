import matplotlib.pyplot as plt
import matplotlib
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import cv2
import glob
from sklearn import metrics
from sklearn.neural_network import MLPClassifier # import Multilayer Perceptron classifier
from sklearn.naive_bayes import GaussianNB # import Naive Bayes classifier
from sklearn import svm # import Support Vector Machine classifier

# get haralick features function
def extract_features(img):
    # create glcm of an image (distance=[1,2], angles=[0,45,90,135])
    glcm = greycomatrix(img, [1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    contrast = greycoprops(glcm, 'contrast')
    energy = greycoprops(glcm, 'energy')
    homogeneity = greycoprops(glcm, 'homogeneity')
    dissimilarity = greycoprops(glcm, 'dissimilarity')

    # compute entropy
    # p = np.zeros((256, 256, 2, 4)) # probability matrix
    # entropy = np.zeros((2,4)) # entropy matrix
    # for d in range(2):
    #     for theta in range(4):
    #         p[:,:,d,theta] = glcm[:,:,d:theta]/glcm[:,:,d:theta].sum()
    #         entropy[d,theta] = sum([
    #             x*(-np.log(x)) if x!=0 else 0 for x in p[:,:,d,theta].flatten()
    #         ])

    # return np.array([contrast, energy, homogeneity, dissimilarity, entropy])
    return np.array([contrast, energy, homogeneity, dissimilarity])

# store image folder into variable
path_normal_train = glob.glob("chest_xray_dataset\\train\\NORMAL\\*.jpeg")
path_pneumonia_train = glob.glob("chest_xray_dataset\\train\\PNEUMONIA\\*.jpeg")

path_normal_test = glob.glob("chest_xray_dataset\\test\\NORMAL\\*.jpeg")
path_pneumonia_test = glob.glob("chest_xray_dataset\\test\\PNEUMONIA\\*.jpeg")

# array to store image files
normal_img_train = []
pneumonia_img_train = []

normal_img_test = []
pneumonia_img_test = []

# read images into array
print('Loading train images . . .')
for img in path_normal_train:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    normal_img_train.append(n)

for img in path_pneumonia_train:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    pneumonia_img_train.append(n)

print('Loading test images . . .')
for img in path_normal_test:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    normal_img_test.append(n)

for img in path_pneumonia_test:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    pneumonia_img_test.append(n)

X_train = []
y_train = []

X_test = []
y_test = []

############################ HARALICK FEATURES OF TRAIN DATA ############################
# extract haralick features of normal chest
for img in normal_img_train:
    img_class = 0 # normal
    features = extract_features(img)
    X_train.append(features)
    y_train.append(img_class)

# extract haralick features of pneumonia chest
for img in pneumonia_img_train:
    img_class = 1 # pneumonia
    features = extract_features(img)
    X_train.append(features)
    y_train.append(img_class)

X_train = np.array(X_train).reshape(((len(normal_img_train) + len(pneumonia_img_train)),32)) # (len(normal_img_test) + (len(pneumonia_img_test) = number of images trained, 40 = distances * orientations * number of features

############################ HARALICK FEATURES OF TEST DATA ############################
# extract haralick features of normal chest
for img in normal_img_test:
    img_class = 0 # normal
    features = extract_features(img)
    X_test.append(features)
    y_test.append(img_class)

# extract haralick features of pneumonia chest
for img in pneumonia_img_test:
    img_class = 1 # pneumonia
    features = extract_features(img)
    X_test.append(features)
    y_test.append(img_class)

X_test = np.array(X_test).reshape((len(normal_img_test) + (len(pneumonia_img_test)),32)) # (len(normal_img_test) + (len(pneumonia_img_test) = number of images trained, 40 = distances * orientations * number of features

# select classifier
clf_sel = input('Select classifier: 1=MLP, 2=NBC, 3=SVM: ')
# selecting and generating the model
if clf_sel==1:
    clf = MLPClassifier(solver='adam', alpha=0.0001,hidden_layer_sizes=(20, 30),max_iter=600)
elif clf_sel==2:
    clf = GaussianNB()
elif clf_sel==3:
    clf = svm.SVC()
else: 
    clf = MLPClassifier(solver='adam', alpha=0.0001,hidden_layer_sizes=(20, 30),max_iter=600)

# train the model
clf.fit(X_train, y_train)
# results
print('############################ RESULTS ############################')
# predict the response
y_pred = clf.predict(X_test)
# accuracy score
print('- Accuracy: ', metrics.accuracy_score(y_test, y_pred))
# precision score
print('- Precision: ', metrics.recall_score(y_test, y_pred))
# summary
print("- Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print('- Test: ', y_test)
print('- Predicted: ', y_pred)
print('############################ END OF RESULTS ############################')