import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import cv2
import glob
from sklearn import metrics
from sklearn.neural_network import MLPClassifier # import Multilayer Perceptron classifier
from sklearn.naive_bayes import GaussianNB # import Naive Bayes classifier
from sklearn.svm import SVC,LinearSVC # import Support Vector Machine classifier
from tqdm import tqdm
from yellowbrick.model_selection import learning_curve

# get haralick features function
def extract_features(img):
    # create glcm of an image (distance=[1,2], angles=[0,45,90,135])
    glcm = greycomatrix(img, [1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    energy = greycoprops(glcm, 'energy')
    homogeneity = greycoprops(glcm, 'homogeneity')
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    correlation = greycoprops(glcm, 'correlation')

    # compute entropy
    p = np.zeros((256, 256, 2, 4)) # probability matrix
    entropy = np.zeros((2,4)) # entropy matrix
    for d in range(2):
        for theta in range(4):
            p[:,:,d,theta] = glcm[:,:,d,theta]/glcm[:,:,d,theta].sum()
            entropy[d,theta] = sum([
                x*(-np.log(x)) if x!=0 else 0 for x in p[:,:,d,theta].flatten()
                ])
    return np.array([energy, homogeneity, contrast, dissimilarity, correlation, entropy])

vector = 2 * 4 * 6 # distances * orientations * number of features

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
print('[STATUS] Loading train images...')
for img in tqdm(path_normal_train, ncols=100):
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    normal_img_train.append(n)

for img in path_pneumonia_train:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    pneumonia_img_train.append(n)

print('[STATUS] Loading test images...')
for img in tqdm(path_normal_test, ncols=100):
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    normal_img_test.append(n)

for img in path_pneumonia_test:
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    pneumonia_img_test.append(n)

# empty list to hold feature vectors and train labels
train_features = []
train_labels = []

# empty list to hold feature vectors and test labels
test_features = []
test_labels = []

############################ HARALICK FEATURES OF TRAIN DATA ############################
# extract haralick features of normal chest
print('[STATUS] Extracting haralick features of normal chests from train data...')
for img in tqdm(normal_img_train, ncols=100):
        img_class = 0 # normal
        features = extract_features(img)
        train_features.append(features)
        train_labels.append(img_class)

# extract haralick features of pneumonia chest
print('[STATUS] Extracting haralick features of pneumonia chests from train data...')
for img in tqdm(pneumonia_img_train, ncols=100):
    img_class = 1 # pneumonia
    features = extract_features(img)
    train_features.append(features)
    train_labels.append(img_class)

total_train = len(normal_img_train) + len(pneumonia_img_train) # total train images
train_features = np.array(train_features).reshape(total_train, vector) 

############################ HARALICK FEATURES OF TEST DATA ############################
# extract haralick features of normal chest
print('[STATUS] Extracting haralick features of normal chests from test data...')
for img in tqdm(normal_img_test, ncols=100):
    img_class = 0 # normal
    features = extract_features(img)
    test_features.append(features)
    test_labels.append(img_class)

# extract haralick features of pneumonia chest
print('[STATUS] Extracting haralick features of pneumonia chests from test data...')
for img in tqdm(pneumonia_img_test, ncols=100):
    img_class = 1 # pneumonia
    features = extract_features(img)
    test_features.append(features)
    test_labels.append(img_class)

total_test = len(normal_img_test) + len(pneumonia_img_test) # total test images
test_features = np.array(test_features).reshape(total_test, vector)

for i in range(3):
    # select classifier
    clf_sel = i 
    if clf_sel==0:
        clf = MLPClassifier(solver='adam', 
                            alpha=0.0001,
                            hidden_layer_sizes=(20, 30),
                            max_iter=600)
        s = 'MULTILAYER PERCEPTRON CLASSIFIER'
    elif clf_sel==1:
        clf = GaussianNB()
        s = 'NAIVE BAYES CLASSIFIER'
    elif clf_sel==2:
        clf = LinearSVC()
        s = 'SUPPORT VECTOR MACHINE CLASSIFIER'

    # train the model
    print('[STATUS] Training...')
    clf.fit(train_features, train_labels)
    # predict the response
    pred_labels = clf.predict(test_features)

    # results
    # determine number of normal/pneumonia images in test data
    normal_test = 0
    pneumonia_test = 0

    for j in test_labels:
        if j==0:
            normal_test += 1
        else:
            pneumonia_test += 1

    # determine number of correctly predicted images
    normal_predicted_correctly = 0
    pneumonia_predicted_correctly = 0

    for k in range(0, len(test_labels)):
        if test_labels[k] == pred_labels[k] == 0:
            normal_predicted_correctly += 1
        elif test_labels[k] == pred_labels[k] == 1:
            pneumonia_predicted_correctly += 1

    # summary
    print('################# %s RESULTS #################' % s)
    print('- No. of normal samples in test samples: ', normal_test)
    print('- No. of normal samples predicted correctly: ', normal_predicted_correctly)
    print('- No. of normal samples predicted incorrectly: ', normal_test - normal_predicted_correctly)
    print('- No. of pneumonia samples in test samples: ', pneumonia_test)
    print('- No. of pneumonia samples predicted correctly: ', pneumonia_predicted_correctly)
    print('- No. of pneumonia samples predicted incorrectly: ', pneumonia_test - pneumonia_predicted_correctly)
    print("- No. of mislabeled samples out of a total %d samples : %d" % (test_features.shape[0], (test_labels != pred_labels).sum()))
    print('- Accuracy: ', metrics.accuracy_score(test_labels, pred_labels)) # accuracy score
    print('- Test: ', test_labels)
    print('- Predicted: ', pred_labels)
    print('############################ END OF RESULTS ############################')
    # confusion matrix
    # cm = metrics.confusion_matrix(test_labels, pred_labels)
    # cmd = (metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)).plot()
    # plt.savefig('ConfusionMatrix_%s' % s)
    print(learning_curve(clf, train_features, train_labels, cv=10, scoring='accuracy'))
