import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
n_neighbors = 6
knn_metric = 'euclidean' # euclidean, manhattan, chebyshev, minkowski, bhatt
# KMeans Clusterer Options
used_kmeans = "mini_batch" # mini_batch, nlkt
distance_kmeans = "corr" # Only for nlkt

#Definitions and implementations
cv2distances = {"corr": cv2.HISTCMP_CORREL,
                "chi": cv2.HISTCMP_CHISQR_ALT,
                "inter": cv2.HISTCMP_INTERSECT,
                "bhatt": cv2.HISTCMP_BHATTACHARYYA,
                "chialt": cv2.HISTCMP_CHISQR_ALT,
                "kl_div": cv2.HISTCMP_KL_DIV
                }
dist_name_list = ["euclidean", "manhattan", "chebyshev", "minkowski", "hamming", "canberra", "braycurtis"] + list(
    cv2distances.keys())

def get_dist_func(name, force_function=False):
    if name in cv2distances.keys():
        def func(x1, x2):
            print("X1:", x1.shape, type(x1))
            print("X2:", x1.shape, type(x2))
            dist = cv2.compareHist(x1, x2, cv2distances[name])
            print(dist)
            return dist

        KNN_METRIC = func

        # KNN_METRIC = lambda x1, x2: cv2.compareHist(x1, x2, cv2distances[name])
    elif not force_function:
        KNN_METRIC = name
    else:
        dist = DistanceMetric.get_metric(name)
        distlambda = lambda x, y: dist.pairwise((x, y))
        KNN_METRIC = distlambda

    return KNN_METRIC

def histogram_intersection(data_1, data_2):
    """
    Generalized histogram intersection kernel
        K(x, y) = SUM_i min(|x_i|^alpha, |y_i|^alpha)
    as defined in
    "Generalized histogram intersection kernel for image recognition"
    Sabri Boughorbel, Jean-Philippe Tarel, Nozha Boujemaa
    International Conference on Image Processing (ICIP-2005)
    http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icip05.pdf
    """
    alpha = 1
    data_1 = np.abs(data_1) ** alpha
    data_2 = np.abs(data_2) ** alpha
    kernel = np.zeros((data_1.shape[0], data_2.shape[0]))

    for d in range(data_1.shape[1]):
        column_1 = data_1[:, d].reshape(-1, 1)
        column_2 = data_2[:, d].reshape(-1, 1)
        kernel += np.minimum(column_1, column_2.T)

    return kernel

def select_svm_kernel(knn_metric):
    if knn_metric in ['linear', 'rbf']:
        return knn_metric
    elif knn_metric == 'hist_intersection':
        return histogram_intersection
