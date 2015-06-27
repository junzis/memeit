import os
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import morphology
from skimage import measure
from skimage import transform

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
            newimg = transform.resize(subimg, (40, 30))
            try:
                uid = str(uuid.uuid1())
                io.imsave( 'data/tmp/output/' + uid + '.jpg', newimg)
            except:
                pass    
    return

def process_image(imgfile):
    '''read image in grey mode'''
    image = img0 = io.imread(imgfile, as_grey=True)

    '''pre-process image'''
    # keep only white part (letters are usually white)
    image = img1 = morphology.white_tophat(image, morphology.disk(15))
    # convert to pure black-white
    image = img2 = np.where(image > 0.9, 0.0, 1.0)

    '''Image segmentation'''
    contours = measure.find_contours(image, 0.8)

    '''save contour images for training'''
    save_contour_images(image, contours)

    '''Display the image and plot all contours found'''
    fig, ax = plt.subplots()
    ax.imshow(img0, interpolation='nearest', cmap=plt.cm.gray)

    # plot contours
    nletters = 0
    for n, contour in enumerate(contours):
        if check_letter(contour):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            nletters += 1

    print 'number of letters:', nletters
    plt.show()


#process_image('data/memes/blb-1.jpg')

for imgfile in os.listdir('data/memes/'):
    print 'Processing image: '+imgfile
    process_image('data/memes/'+imgfile)
