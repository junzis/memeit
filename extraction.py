import os
import uuid
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import morphology
from skimage import measure

import pytesseract
from PIL import Image

def check_letter(contour):
        xmax = max(contour[:, 1])
        xmin = min(contour[:, 1])
        ymax = max(contour[:, 0])
        ymin = min(contour[:, 0])
        dx = abs(xmax-xmin)
        dy = abs(ymax-ymin)
        if dx>10 and dy>10:
            return True
        else:
            return False

def save_contour_images(image, contours):
    """save the image within contours for training"""
    for n, contour in enumerate(contours):
        xmax = max(contour[:, 1])
        xmin = min(contour[:, 1])
        ymax = max(contour[:, 0])
        ymin = min(contour[:, 0])

        if check_letter(contour):
            subimg =  image[ymin:ymax, xmin:xmax]
            try:
                uid = str(uuid.uuid1())
                io.imsave( 'data/output/' + uid + '.jpg', subimg)
            except:
                pass    
    return

def process_image(imgfile):
    '''read image in grey mode'''
    image = origimg = io.imread(imgfile, as_grey=True)

    '''pre-process image'''
    # keep only white part (letters are usually white)
    image = morphology.white_tophat(image, morphology.disk(15))
    # convert to pure black-white
    image = np.where(image > 0.9, 0.0, 1.0)

    '''Image segmentation'''
    contours = measure.find_contours(image, 0.8)

    '''save contour images for training'''
    save_contour_images(image, contours)

    '''Display the image and plot all contours found'''
    fig, ax = plt.subplots()
    ax.imshow(origimg, interpolation='nearest', cmap=plt.cm.gray)

    # plot contours
    nletters = 0
    for n, contour in enumerate(contours):
        if check_letter(contour):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            nletters += 1

    print 'number of letters:', nletters
    plt.show()


# for imgfile in os.listdir('data/memes/'):
#   process_image('data/memes/'+imgfile)
process_image('data/memes/duck-1.jpg')