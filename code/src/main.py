import cv2
import numpy as np
import pickle
from tqdm import tqdm
print("hola")
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer, StandardScaler

from assessment import showConfusionMatrix
from descriptors import get_keypoints, get_descriptors, get_bag_of_words, get_visual_words
from classifiers import get_dist_func

#Train parameters
kp_detector = 'sift' # sift dense
desc_type = 'sift' # sift
n_descriptors = 600

#visual words pyramids
mode_bagofWords = 'pyramids'

#pyramids params
levels_pyramid = 2
codebook_sizes = [128]
normalize_level_vw = True
scaleData_level_vw = False

classif_type  =  'svm' # knn svm
knn_metric = 'euclidean'
kernel_type = 'hist_intersection' #'rbf' or 'hist_intersection'
save_trainData = False

#data normalization and scalation
normalize = True
scaleData = False

#Dense Params
stepValue = 10
scale_mode = "gauss" # random, uniform, gauss

#uniform 
maxScale = 15
minScale = 7
#gauss
mean = 15
desvt= 7


def save_data(object, filename):
    filehandler = open(filename, 'wb')
    pickle.dump(object, filehandler)


def load_data(filename):
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)

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

def select_svm_kernel(kernel_type):
    if kernel_type in ['linear', 'rbf']:
        return kernel_type
    elif kernel_type == 'hist_intersection':
        return histogram_intersection

if __name__ == '__main__':

    # Load train and test files
    total_train_images_filenames = pickle.load(open('./train_images_filenames.dat', 'rb'))
    total_test_images_filenames = pickle.load(open('./test_images_filenames.dat', 'rb'))
    total_train_labels = pickle.load(open('./train_labels.dat', 'rb'))
    total_test_labels = pickle.load(open('./test_labels.dat', 'rb'))

    # Split train dataset for cross-validation
    cv = StratifiedKFold(n_splits=2)

    for codebook_size in codebook_sizes:
        accumulated_accuracy=[]
        for train_index, val_index in cv.split(total_train_images_filenames, total_train_labels):
            # train_index = train_index[:10]
            # val_index = val_index[:10]
            train_filenames = [total_train_images_filenames[index] for index in train_index]
            train_labels = [total_train_labels[index] for index in train_index]
            val_filenames = [total_train_images_filenames[index] for index in val_index]
            val_labels = [total_train_labels[index] for index in val_index]

            # TRAIN CLASSIFIER

            keypoint_list = []
            train_desc_list = []
            train_label_per_descriptor = []

            for filename, labels in zip(tqdm(train_filenames), train_labels):
                ima = cv2.imread(filename)
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                # 1. Detect keypoints (sift or dense)
                kpt = get_keypoints(gray, kp_detector, stepValue, scale_mode, minScale, maxScale, mean, desvt, n_descriptors=n_descriptors)
                # 2. Get descriptors (normal sift or spatial pyramid)
                kpt, des = get_descriptors(gray, kpt, desc_type)
                keypoint_list.append(kpt)
                train_desc_list.append(des)
                train_label_per_descriptor.append(labels)
                #separar el descriptor en diferentes listas (?)

            D = np.vstack(train_desc_list)

            # 4. Create codebook and fit with train dataset
            codebook, visual_words = get_bag_of_words(levels_pyramid, mode_bagofWords, D, train_desc_list, keypoint_list, codebook_size, normalize_level_vw=normalize_level_vw, scaleData_level_vw=scaleData_level_vw)

            # 3. Normalize and scale descriptors
            if(normalize):
                norm_model = Normalizer().fit(visual_words) #l2 norm by default
                visual_words = norm_model.fit_transform(visual_words)

            #scaling
            if(scaleData):
                scale_model = StandardScaler().fit(visual_words)
                visual_words = scale_model.fit_transform(visual_words)


            # 5. Train classifier
            model = None
            if classif_type == 'knn':
                model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric=get_dist_func(knn_metric))
                model.fit(visual_words, train_labels)
            elif classif_type == 'svm':
                svm_kernel = select_svm_kernel(kernel_type)
                model = SVC(C=1.0, kernel=svm_kernel, degree=3, gamma='auto', shrinking=False, probability=False, tol=0.001, max_iter=-1)
                model.fit(visual_words, train_labels)
            else:
                raise (NotImplemented("classif_type not implemented or not recognized:" + str(classif_type)))

            # 6. Save/load data in pickle
            if save_trainData:
                save_data(codebook, "codebook.pkl")
                save_data(visual_words, "visual_words.pkl")
            else:
                pass
                #codebook = load_data("codebook.pkl")
                #visual_words = load_data("visual_words.pkl")

            # VALIDATE CLASSIFIER WITH CROSS-VALIDATION DATASET

            if(mode_bagofWords == 'all'):
                visual_words_test = np.zeros((len(val_filenames), codebook_size), dtype=np.float32)
            if(mode_bagofWords == 'pyramids'):
                len_vw = 0
                for i in range(0, levels_pyramid):
                    len_vw += 2**(2*i)
                len_vw *= codebook_size
                visual_words_test = np.zeros((len(val_filenames), len_vw), dtype=np.float32)

            for i in tqdm(range(len(val_filenames))):
                filename = val_filenames[i]
                ima = cv2.imread(filename)
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                kpt = get_keypoints(gray, kp_detector)
                kpt, des = get_descriptors(gray, kpt, desc_type)

                _, visual_words_test[i,:] = get_visual_words(levels_pyramid, mode_bagofWords, codebook, des, kpt, codebook_size)

            if(normalize):
                visual_words_test = norm_model.transform(visual_words_test)
            if(scaleData):
                visual_words_test = scale_model.transform(visual_words_test)

            # ASSESSMENT OF THE CLASSIFIER
            accuracy = 100 * model.score(visual_words_test, val_labels)
            accumulated_accuracy.append(accuracy)
        print("codebook size: ", codebook_size, " accuracy: ", np.sum(accumulated_accuracy)/len(accumulated_accuracy))
            # Show Confusion Matrix
            # showConfusionMatrix(dist_name_list, conf_mat_list, labels_names)
