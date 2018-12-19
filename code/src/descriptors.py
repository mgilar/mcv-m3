import numpy as np
import random
import cv2
from sklearn.cluster import MiniBatchKMeans
from nltk.cluster.kmeans import KMeansClusterer

from classifiers import get_dist_func

#Descriptor parameters

dense_descriptors = True
#SIFT Params
maxScale = 15
minScale = 7
n_descriptors = 600
codebook_size = 128
#Dense Params
if dense_descriptors:
    stepValue = 10
    mean = 15
    desvt= 7
    scale_mode = "gauss" # random, uniform, gauss


class KMeansDistances(KMeansClusterer):
    """
    KMeansClusterer adaptation to use same sintax as MiniBatchKMeans
    Implementation of KMeans that allows using distances.
    """

    def __init__(self, *args, **kwargs):
        args += (nltk.cluster.util.cosine_distance,)
        super().__init__(*args, **kwargs)

    def fit(self, data):
        return self.cluster(data, True, trace=True)

    def predict(self, vector):
        return self.classify_vectorspace(vector)

def dense_keypoints(img, step, scaleMin, scaleMax):
    keypoints = []
    heigh, width = img.shape
    for i in range(0, heigh, step):
        for j in range(0, width, step):
            if (scale_mode == "random"):
                scale = np.random.rand()
            elif (scale_mode == "uniform"):
                scale = random.uniform(scaleMin, scaleMax)  # maybe another type of randomness?
            elif (scale_mode == "gauss"):
                scale = abs(random.gauss(mean, desvt))
            keypoints.append(cv2.KeyPoint(j, i, scale))
    return keypoints

def get_keypoints(im, detector_type):
    if (detector_type == 'dense'):
        kpt = dense_keypoints(im, stepValue, maxScale, minScale)
    elif (detector_type == 'sift'):
        detectorObject = cv2.xfeatures2d.SIFT_create(nfeatures=n_descriptors)
        kpt = detectorObject.detect(im)
    else:
        raise (NotImplemented("detector_type not implemented or not recognized:" + str(detector_type)))
    return kpt

def get_descriptors(im, kpt, descriptor_type):
    if descriptor_type == 'sift':
        detectorObject = cv2.xfeatures2d.SIFT_create()
        des = detectorObject.compute(im, kpt)
    elif descriptor_type == 'spatial_pyramid':
        #TODO
        des = []
    else:
        raise (NotImplemented("descriptor_type not implemented or not recognized:" + str(descriptor_type)))
    return des

def get_bag_of_words(D, desc_list, codebook_size, used_kmeans='mini_batch'):
    # We now compute a k-means clustering on the descriptor space
    reassignment_ratio = 10 ** -4
    if (used_kmeans == "mini_batch"):
        codebook = MiniBatchKMeans(n_clusters=codebook_size, verbose=False, batch_size=codebook_size * 20,
                                   compute_labels=False, reassignment_ratio=reassignment_ratio, random_state=42)
    elif (used_kmeans == "nlkt"):
        DIST = get_dist_func('corr', True)
        # codebook = KMeansDistances(codebook_size, DIST, repeats=10)
        codebook = KMeansDistances(codebook_size, repeats=10)
    else:
        raise (ValueError("KMeans not recognized"))
    codebook.fit(D)
    # And, for each train image, we project each keypoint descriptor to its closest visual word.
    # We represent each of the images with the frequency of each visual word.
    visual_words = np.zeros((len(desc_list), codebook_size), dtype=np.float32)
    for i in range(len(desc_list)):
        words = codebook.predict(desc_list[i])
        visual_words[i, :] = np.bincount(words, minlength=codebook_size)
    return codebook, visual_words


