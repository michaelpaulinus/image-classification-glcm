# Image Classification using GLCM
 This project uses Haralick features from a Gray Level Co-occurrence Matrix (GLCM) to classify X-ray scans of a chest as that of a chest that is normal or having pneumonia.


The dataset used for the project was taken from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. It consisted of a total of 5276 images for training and 624 images for testing.

![image](https://user-images.githubusercontent.com/56998775/139958688-120f28d0-db07-4369-8797-3f52ca3b42b6.png)

 The project used three different classifier models:
 1) Multilayer Perceptron Based Classifier
 2) Naive Bayes Classifier
 3) Support Vector Machines Based Classifier
 
 The accuracies achieved for the three classifiers were:
 1) Multilayer Perceptron Based Classifier - 50%
 2) Naive Bayes Classifier - 50%
 3) Support Vector Machines Based Classifier - 50%

 # Installing OpenCV
 ```bash
 pip install opencv-python
 ```
 # Installing Numpy
 ```bash
 pip install numpy
 ```
 # Installing scikit-image
 ```bash
 pip install -U scikit-image
 ```
 # Installing scikit-learn
 ```bash
 pip install -U scikit-learn
 ```
 # Installing tqdm
 ```bash
 pip install tqdm
 ```
