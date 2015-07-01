import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import morphology
from skimage import measure
from skimage import transform

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cluster import MeanShift, estimate_bandwidth

from wordsplit import infer_spaces

def check_letter(xmax, xmin, ymax, ymin):
    """Check if an contour seems like letter"""
    dx = abs(xmax-xmin)
    dy = abs(ymax-ymin)
    scale = float(dx)/float(dy)

    if scale > 0.2 and scale < 1.2 and dx>4 and dy>10:
        return True
    else:
        return False

def process_image(imgfile):
    ltrimgs = []
    ltrcenters = []

    '''read image in grey mode'''
    image = img0 = io.imread(imgfile, as_grey=True)

    '''pre-process image'''
    # keep only white part (letters are usually white)
    image = img1 = morphology.white_tophat(image, morphology.disk(15))
    # convert to pure black-white
    image = img2 = np.where(image > 0.9, 0.0, 1.0)

    '''Image segmentation'''
    contours = measure.find_contours(image, 0.8)

    '''Process letters'''    
    for n, contour in enumerate(contours):
        xmax = max(contour[:, 1])
        xmin = min(contour[:, 1])
        ymax = max(contour[:, 0])
        ymin = min(contour[:, 0])

        if check_letter(xmax, xmin, ymax, ymin):
            subimg =  image[ymin:ymax, xmin:xmax] * 255  # convert to greyscalel
            ltrimg = transform.resize(subimg, (40, 30))
            ltrimgs.append(ltrimg.reshape([1, -1])[0].tolist())

            x = int(abs(xmax+xmin) / 2)
            y = int(abs(ymax+ymin) / 2)
            ltrcenters.append([x, y])
            
    return (ltrimgs, ltrcenters)



#########################################################
## Loading classifiers from pickle
estimator = pickle.load(open('data/clfs/estimator.pkl', 'rb'))
pca = pickle.load(open('data/clfs/pca.pkl', 'rb'))

#########################################################
## Test with pickled data
# X_train,  X_test, y_train, y_test = pickle.load(open('data/pkls/dataset.pkl'))
# y_pred = estimator.predict(pca.transform(X_test))
# print(classification_report(y_test, y_pred))

def read_meme(memeimg, draw=False):
    # Analyse MEME image
    ltrimgs, imgcenters = process_image(memeimg)
    ltrcenters = np.asarray(imgcenters)
    letters = []
    probs = []

    results = estimator.predict_proba(pca.transform(ltrimgs))
    classes = estimator.classes_

    for r in results:
        i = np.argsort(r)[-1]
        letters.append(classes[i])
        probs.append(r[i])

    letters = np.asarray(letters)
    probs = np.asarray(probs)

    # apply clustering methord to align letters to rows
    X = np.array(zip(ltrcenters[:, 1], np.zeros(len(ltrcenters))), dtype=np.int)
    ms = MeanShift(bandwidth=5, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    for k in range(n_clusters_):
        members = (labels == k)
        ltrcenters[members, 1] = int(cluster_centers[k][0])

    ids = np.lexsort((ltrcenters[:,0], ltrcenters[:,1]))

    # print zip(letters[ids], ltrcenters[ids, 0].tolist())

    # get the text output, in lowercase
    letters_sort = letters[ids]
    probs_sort = probs[ids]
    words = ''
    for l in zip(letters_sort, probs_sort):
        if l[1] > 0.4:      # only keep prob > 40%
            words += l[0]
    print infer_spaces(words.lower())

    # draw the letters
    if draw:
        plt.plot(ltrcenters[:, 0], ltrcenters[:, 1], '.', color='blue')
        for l in zip(imgcenters, letters, probs):
            if l[2] > 0.4:
                plt.text(l[0][0], l[0][1], l[1], color='black', weight='bold')
                plt.text(l[0][0], l[0][1]+15, "{0:.2f}".format(l[2]), color='green', size=10)
            else:
                plt.text(l[0][0], l[0][1], l[1], color='black', alpha=0.5)
                plt.text(l[0][0], l[0][1]+15, "{0:.2f}".format(l[2]), color='red', size=10, alpha=0.5)
        plt.gca().invert_yaxis()
        plt.show()

if __name__ == '__main__':
    read_meme('data/memes/ida-4.jpg', draw=True)
    # for d in os.listdir('data/meme-themes/'):
    #     print '======================'
    #     print d
    #     print '======================'
    #     for f in os.listdir('data/meme-themes/'+d+'/'):
    #         read_meme('data/meme-themes/' + d + '/' + f, draw=False)
