import cv2
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
n_neighbors = 6
knn_metric = 'minkowski' # euclidean, manhattan, chebyshev, minkowski, bhatt
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