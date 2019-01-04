import cv2
import numpy as np
import pickle
from tqdm import tqdm
import random

from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer, StandardScaler

from assessment import showConfusionMatrix
from descriptors import get_bag_of_words, get_visual_words, compute_descriptors, get_pyramid_visual_word_len, select_descriptors
from classifiers import get_dist_func, select_svm_kernel

def save_data(object, filename):
    filehandler = open(filename, 'wb')
    pickle.dump(object, filehandler)


def load_data(filename):
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)

class Classification(object):
    def __init__(self):
        # Load train and test files
        self.total_train_images_filenames = pickle.load(open('./train_images_filenames.dat', 'rb'))
        self.total_test_images_filenames = pickle.load(open('./test_images_filenames.dat', 'rb'))
        self.total_train_labels = pickle.load(open('./train_labels.dat', 'rb'))
        self.total_test_labels = pickle.load(open('./test_labels.dat', 'rb'))

        #Train parameters
        self.kp_detector = 'sift' # sift dense
        self.desc_type = 'sift' # sift
        self.n_descriptors = 600
        self.split_num = 2

        #visual words pyramids
        self.mode_bagofWords = 'pyramids'
        self.reduce_num_of_features = False
        self.features_per_img = 100

        #pyramids params
        self.levels_pyramid = 2
        self.codebook_size = 128
        self.normalize_level_vw = True
        self.scaleData_level_vw = False

        self.classif_type  =  'svm' # knn svm
        self.knn_metric = 'euclidean'
        self.svm_metric = 'hist_intersection' #'rbf' or 'hist_intersection'
        self.save_trainData = False

        #svm 
        self.C=1.0
        self.degree=3
        self.gamma='auto'

        #data normalization and scalation
        self.normalize = True
        self.scaleData = False

        #Dense Params
        self.stepValue = 10
        self.scale_mode = "gauss" # random, uniform, gauss

        #uniform 
        self.maxScale = 15
        self.minScale = 7
        #gauss
        self.mean = 15
        self.desvt= 7

    def compute(self):
        # Split train dataset for cross-validation
        cv = StratifiedKFold(n_splits=self.split_num)

        accumulated_accuracy=[]        
        for train_index, val_index in cv.split(self.total_train_images_filenames, self.total_train_labels):
            train_index = train_index[:200]
            val_index = val_index[:200]
            train_filenames = [self.total_train_images_filenames[index] for index in train_index]
            train_labels = [self.total_train_labels[index] for index in train_index]
            val_filenames = [self.total_train_images_filenames[index] for index in val_index]
            val_labels = [self.total_train_labels[index] for index in val_index]

            # TRAIN CLASSIFIER
            keypoint_list = []
            train_desc_list = []
            train_label_per_descriptor = []

            for filename, labels in zip(tqdm(train_filenames, desc="train descriptors"), train_labels):
                ima = cv2.imread(filename)
                kpt, des = compute_descriptors(ima, self.kp_detector, self.desc_type, self.stepValue, self.scale_mode, self.minScale, self.maxScale, self.mean, self.desvt, self.n_descriptors)
                keypoint_list.append(kpt)
                train_desc_list.append(des)
                train_label_per_descriptor.append(labels)
            
            D = np.vstack(train_desc_list)

            # reducing the number of descriptors used in bag of words
            if(self.reduce_num_of_features):
                selected_index = select_descriptors(train_desc_list, self.features_per_img)
                D = D[selected_index]

            # 3. Create codebook and fit with train dataset
            codebook, visual_words = get_bag_of_words(self.levels_pyramid, self.mode_bagofWords, D, train_desc_list, keypoint_list, self.codebook_size, normalize_level_vw=self.normalize_level_vw, scaleData_level_vw=self.scaleData_level_vw)

            # 4. self.normalize and scale descriptors
            if(self.normalize):
                norm_model = Normalizer().fit(visual_words) #l2 norm by default
                visual_words = norm_model.fit_transform(visual_words)

            #scaling
            if(self.scaleData):
                scale_model = StandardScaler().fit(visual_words)
                visual_words = scale_model.fit_transform(visual_words)


            # 5. Train classifier
            model = None
            if self.classif_type == 'knn':
                model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric=get_dist_func(self.knn_metric))
                model.fit(visual_words, train_labels)
            elif self.classif_type == 'svm':
                svm_kernel = select_svm_kernel(self.svm_metric)
                model = SVC(C=self.C, kernel=svm_kernel, degree=self.degree, gamma=self.gamma, shrinking=False, probability=False, tol=0.001, max_iter=-1)
                model.fit(visual_words, train_labels)
            else:
                raise (NotImplemented("self.classif_type not implemented or not recognized:" + str(self.classif_type)))

            # 6. Save/load data in pickle
            if self.save_trainData:
                save_data(codebook, "codebook.pkl")
                save_data(visual_words, "visual_words.pkl")
            else:
                pass
                #codebook = load_data("codebook.pkl")
                #visual_words = load_data("visual_words.pkl")

            # VALIDATE CLASSIFIER WITH CROSS-VALIDATION DATASET

            if(self.mode_bagofWords == 'all'):
                visual_words_test = np.zeros((len(val_filenames), self.codebook_size), dtype=np.float32)
            if(self.mode_bagofWords == 'pyramids'):
                len_vw = get_pyramid_visual_word_len(self.levels_pyramid,self.codebook_size)
                visual_words_test = np.zeros((len(val_filenames), len_vw), dtype=np.float32)

            for i in tqdm(range(len(val_filenames)), desc="test descriptors"):
                filename = val_filenames[i]
                ima = cv2.imread(filename)
                kpt, des = compute_descriptors(ima, self.kp_detector, self.desc_type, self.stepValue, self.scale_mode, self.minScale, self.maxScale, self.mean, self.desvt, self.n_descriptors )

                _, visual_words_test[i,:] = get_visual_words(self.levels_pyramid, self.mode_bagofWords, codebook, des, kpt, self.codebook_size,  normalize_level_vw=self.normalize_level_vw, scaleData_level_vw=self.scaleData_level_vw)

            if(self.normalize):
                visual_words_test = norm_model.transform(visual_words_test)
            if(self.scaleData):
                visual_words_test = scale_model.transform(visual_words_test)

            # ASSESSMENT OF THE CLASSIFIER
            accuracy = 100 * model.score(visual_words_test, val_labels)
            accumulated_accuracy.append(accuracy)

            # Show Confusion Matrix
            # showConfusionMatrix(dist_name_list, conf_mat_list, labels_names)

            return np.sum(accumulated_accuracy)/len(accumulated_accuracy)

if __name__ == "__main__":
    imageclassifier = Classification()
    accuracy = imageclassifier.compute()
    print(" accuracy: ", accuracy)

