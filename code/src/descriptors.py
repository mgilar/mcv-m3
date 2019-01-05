import numpy as np
import random
import cv2
from tqdm import tqdm 
from sklearn.cluster import MiniBatchKMeans
from nltk.cluster.kmeans import KMeansClusterer

from classifiers import get_dist_func

from histograms import accBackpropagationHistograms
from sklearn.preprocessing import Normalizer, StandardScaler


#pyramid size
image_width = 256
image_height = 256

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

def get_pyramid_visual_word_len(levels_pyramid, codebook_size):
    len_vw = 0
    for i in range(0, levels_pyramid):
        len_vw += 2**(2*i)
    len_vw *= codebook_size
    return len_vw

def dense_keypoints(img, step, scale_mode, scaleMin, scaleMax, mean, desvt ):
    keypoints = []
    heigh, width = img.shape
    for i in range(0, heigh, step):
        for j in range(0, width, step):
            if (scale_mode == "multiple"):
                scales = [4, 8, 12, 16]
                for scale in scales:
                    keypoints.append(cv2.KeyPoint(j, i, scale))
            else:
                if (scale_mode == "random"):
                    scale = np.random.rand()
                elif (scale_mode == "uniform"):
                    scale = random.uniform(scaleMin, scaleMax)  # maybe another type of randomness?
                elif (scale_mode == "gauss"):
                    scale = abs(random.gauss(mean, desvt))
                keypoints.append(cv2.KeyPoint(j, i, scale))
    return keypoints


def get_keypoints(im, detector_type, step, scale_mode, scaleMin, scaleMax, mean, desvt, n_descriptors=600):
    if (detector_type == 'dense'):
        kpt = dense_keypoints(im, step, scale_mode, scaleMin, scaleMax, mean, desvt )
    elif (detector_type == 'sift'):
        detectorObject = cv2.xfeatures2d.SIFT_create(nfeatures=n_descriptors)
        kpt = detectorObject.detect(im)
    else:
        raise (NotImplemented("detector_type not implemented or not recognized:" + str(detector_type)))
    return kpt

def get_descriptors(im, kpt, descriptor_type):
    if descriptor_type == 'sift':
        detectorObject = cv2.xfeatures2d.SIFT_create()
        kpt,des = detectorObject.compute(im, kpt)
    else:
        raise (NotImplemented("descriptor_type not implemented or not recognized:" + str(descriptor_type)))
    return kpt,des

def select_descriptors(desc_list, descriptors_per_image ):
    selected_index = []
    addition_idx = 0
    for x in desc_list:
        num_samples = min(len(x),descriptors_per_image)
        idx = random.sample(range(0,len(x)), num_samples)
        selected_index.extend(np.add(idx, addition_idx))
        addition_idx += len(x)
    return np.array(selected_index)

def get_bag_of_words(levels_pyramid, mode, D, desc_list, keypoint_list, codebook_size, normalize_level_vw=True, scaleData_level_vw=False, used_kmeans='mini_batch'):
    # We now compute a k-means clustering on the descriptor space
    reassignment_ratio = 10 ** -4
    if (used_kmeans == "mini_batch"):
        codebook = MiniBatchKMeans(n_clusters=codebook_size, verbose=False, batch_size=codebook_size * 20,
                                   compute_labels=False, reassignment_ratio=reassignment_ratio, random_state=42)
    elif (used_kmeans == "nlkt"):
        codebook = KMeansDistances(codebook_size, repeats=10)
    else:
        raise (ValueError("KMeans not recognized"))
    codebook.fit(D)

    if(mode == 'all'):
        # And, for each train image, we project each keypoint descriptor to its closest visual word.
        # We represent each of the images with the frequency of each visual word.
        visual_words = np.zeros((len(desc_list), codebook_size), dtype=np.float32)
        for i in tqdm(range(len(desc_list))):
            words = codebook.predict(desc_list[i])
            visual_words[i, :] = np.bincount(words, minlength=codebook_size)

    elif(mode == 'pyramids'):
        len_vw = 0
        for i in range(0, levels_pyramid):
            len_vw += 2**(2*i)
        len_vw *= codebook_size
        visual_words = np.zeros((len(desc_list), len_vw), dtype=np.float32)
        
        for i in (range(len(desc_list))):
            words = codebook.predict(desc_list[i])
            vw = accBackpropagationHistograms(keypoint_list[i] , words, codebook_size, levels_pyramid-1, image_height, image_width)
            
            #normalization
            if(normalize_level_vw):
                for level in range(0, levels_pyramid):
                    norm_model = Normalizer().fit(vw[level]) #l2 norm by default
                    vw[level] = norm_model.fit_transform(vw[level])

            #scaling
            if(scaleData_level_vw):
                for level in range(0, levels_pyramid):
                    scale_model = StandardScaler().fit(vw[level])
                    vw[level] = scale_model.fit_transform(vw[level])

            for level in range(0, levels_pyramid):
                vw[level] = vw[level][0]
            
            visual_words[i, :] = np.concatenate(vw)

    return codebook, visual_words


def get_visual_words(levels_pyramid, mode, codebook, descriptor, keypoints, codebook_size, normalize_level_vw=True, scaleData_level_vw=False):
    if(mode == 'all'):
        visual_words = np.zeros((1, codebook_size), dtype=np.float32)
        words = codebook.predict(descriptor)
        visual_words[0, :] = np.bincount(words, minlength=codebook_size)
    elif(mode == 'pyramids'):
        len_vw = 0
        for i in range(0, levels_pyramid):
            len_vw += 2**(2*i)
        len_vw *= codebook_size
        visual_words = np.zeros((1, len_vw), dtype=np.float32)
        words = codebook.predict(descriptor)
        vw = accBackpropagationHistograms(keypoints , words, codebook_size, levels_pyramid-1, image_height, image_width)
        
        #normalization
        if(normalize_level_vw):
            for level in range(0, levels_pyramid):
                norm_model = Normalizer().fit(vw[level]) #l2 norm by default
                vw[level] = norm_model.fit_transform(vw[level])

        #scaling
        if(scaleData_level_vw):
            for level in range(0, levels_pyramid):
                scale_model = StandardScaler().fit(vw[level])
                vw[level] = scale_model.fit_transform(vw[level])

        for level in range(0, levels_pyramid):
            vw[level] = vw[level][0]
        visual_words[0, :] = np.concatenate(vw)
    return codebook, visual_words

def compute_descriptors(ima, kp_detector, desc_type, stepValue, scale_mode, minScale, maxScale, mean, desvt, n_descriptors ):
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    # 1. Detect keypoints (sift or dense)
    kpt = get_keypoints(gray, kp_detector, stepValue, scale_mode, minScale, maxScale, mean, desvt, n_descriptors=n_descriptors)
    # 2. Get descriptors (normal sift or spatial pyramid)
    kpt, des = get_descriptors(gray, kpt, desc_type)
    return kpt, des
