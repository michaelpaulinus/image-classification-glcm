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
from sklearn.model_selection import ShuffleSplit

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
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
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
        s = 'MLP'
    elif clf_sel==1:
        clf = GaussianNB()
        s = 'NBC'
    elif clf_sel==2:
        clf = LinearSVC()
        s = 'SVM'

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
    cm = metrics.confusion_matrix(test_labels, pred_labels)
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_).plot()
    plt.savefig('ConfusionMatrix_%s' % s)
    plt.close()
    # learning curve 1
    plot_learning_curve(clf, 'Learning Curve (%s)'%s, train_features, train_labels,  ylim=(0.7, 1.01), cv=10)
    plt.savefig('LearningCurve_%s' % s)
    plt.close()
    # learning curve 2
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(clf, 'Learning Curve (%s)'%s, train_features, train_labels,  ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    plt.savefig('LearningCurve2_%s' % s)
    plt.close()
