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
from sklearn.model_selection import learning_curve

############################ FUNCTIONS ############################
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

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

############################ PROGRAM ############################
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
print('[STATUS] Loading normal train images...')
for img in tqdm(path_normal_train, ncols=100):
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    normal_img_train.append(n)

print('[STATUS] Loading pneumonia train images...')
for img in tqdm(path_pneumonia_train, ncols=100):
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    pneumonia_img_train.append(n)

print('[STATUS] Loading normal test images...')
for img in tqdm(path_normal_test, ncols=100):
    n = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    normal_img_test.append(n)

print('[STATUS] Loading pneumonia test images...')
for img in tqdm(path_pneumonia_test, ncols=100):
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

############################ TRAINING ############################
for i in range(3):
    # select classifier
    clf_sel = i 
    if clf_sel==0:
        clf = MLPClassifier(solver='adam', 
                            alpha=0.0001,
                            hidden_layer_sizes=(20, 30),
                            max_iter=600)
        clf_str = 'MLP'
    elif clf_sel==1:
        clf = GaussianNB()
        clf_str = 'NBC'
    elif clf_sel==2:
        clf = LinearSVC()
        clf_str = 'SVM'

    # train the model
    print('[STATUS] Training...')
    clf.fit(train_features, train_labels)
    # predict the response
    pred_labels = clf.predict(test_features)

    ############################ RESULTS ############################
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
    print('################# %s RESULTS #################' % clf_str)
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
    cm = metrics.confusion_matrix(test_labels, pred_labels)
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_).plot()
    plt.savefig('ConfusionMatrix_%s' % clf_str)
    plt.close()
    # learning curve
    plot_learning_curve(clf, 'Learning Curve (%s)'%clf_str, train_features, train_labels,  ylim=(0.7, 1.01), cv=10)
    plt.savefig('LearningCurve_%s' % clf_str)
    plt.close()