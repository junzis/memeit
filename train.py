import os
import numpy as np
from scipy import misc
from time import time
from matplotlib import pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

import pickle

def read_image(imgfile):
    ''' read a char image, and convert into 1D list
    '''
    img_raw = misc.imread(imgfile)
    img_1d_list = img_raw.reshape([1, -1])[0].tolist()
    return img_1d_list


####################################################################
# Process data in the training dir, prepare it for learning
imgdata = [] 
imglabels = []

for c in os.listdir('data/train'):
    ldir = 'data/train/' + c
    if os.path.isdir(ldir):
        for f in os.listdir(ldir):
            fpath = ldir + '/' + f
            if f.endswith('.jpg'):
                imgdata.append(read_image(fpath))
                imglabels.append(c)

# dataset and label in numpy array
X = np.asarray(imgdata)
y = np.asarray(imglabels)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


####################################################################
# Apply PCA, reduce input dimensions space from 1200 to lower
n_components = 150

t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("PCA done in %0.3fs" % (time() - t0))

t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Data transform done in %0.3fs" % (time() - t0))

# dump data to pickle
pickle.dump([X_train, X_test, y_train, y_test], open('data/pkls/dataset.pkl', 'wb'))
pickle.dump(pca, open('data/clfs/pca.pkl', 'wb'))

####################################################################
# Apply SVM classifier
# Before apply SVC, use a search method to find the best params
#   C - Penalty parameter C of the error term.
#   gamma - Kernel coefficient
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto', probability=True), param_grid)
clf.fit(X_train_pca, y_train)
print("Clf done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

pickle.dump(clf.best_estimator_, open('data/clfs/estimator.pkl', 'wb'))

####################################################################
# Prediction and validation
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# eigenltrs = pca.components_.reshape((n_components, 40, 30))
# plt.imshow(eigenltrs[1], cmap=plt.cm.gray)
# plt.show()
