import cv2
import numpy as np
import pickle
from tqdm import tqdm

from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from assessment import showConfusionMatrix
from descriptors import get_keypoints, get_descriptors, get_bag_of_words
from classifiers import get_dist_func
from sklearn.preprocessing import Normalizer

#Train parameters
kp_detector = 'sift' #sift 
desc_type = 'sift' #pyramid
codebook_sizes = [100,500,700,900,1000]
classif_type  =  'knn' #svm
knn_metric = 'euclidean'
save_trainData = False
normalize = False


def save_data(object, filename):
    filehandler = open(filename, 'wb')
    pickle.dump(object, filehandler)


def load_data(filename):
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)

if __name__ == '__main__':

    # Load train and test files
    total_train_images_filenames = pickle.load(open('./train_images_filenames_unix.dat', 'rb'))
    total_test_images_filenames = pickle.load(open('./test_images_filenames_unix.dat', 'rb'))
    total_train_labels = pickle.load(open('./train_labels_unix.dat', 'rb'))
    total_test_labels = pickle.load(open('./test_labels_unix.dat', 'rb'))

    #Split train dataset for cross-validation
    cv = StratifiedKFold(n_splits=5)

    for codebook_size in codebook_sizes:
        accumulated_accuracy=[]
        for train_index, val_index in cv.split(total_train_images_filenames, total_train_labels):
            train_filenames = [total_train_images_filenames[index] for index in train_index]
            train_labels = [total_train_labels[index] for index in train_index]
            val_filenames = [total_train_images_filenames[index] for index in val_index]
            val_labels = [total_train_labels[index] for index in val_index]

            # TRAIN CLASSIFIER

            train_desc_list = []
            Train_label_per_descriptor = []

            for filename, labels in zip(tqdm(train_filenames), train_labels):
                ima = cv2.imread(filename)
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                # 1. Detect keypoints (sift or dense)
                kpt = get_keypoints(gray, kp_detector)
                # 2. Get descriptors (normal sift or spatial pyramid)
                kpt, des = get_descriptors(gray, kpt, desc_type)
                train_desc_list.append(des)
                Train_label_per_descriptor.append(labels)
                #separar el descriptor en diferentes listas (?)

            D = np.vstack(train_desc_list)

            # 4. Create codebook and fit with train dataset
            codebook, visual_words = get_bag_of_words(D, train_desc_list, codebook_size)

            # 3. Normalize descriptors TODO
            if(normalize):
                transformer = Normalizer().fit(visual_words)
                visual_words = transformer.fit_transform(visual_words)

            # 5. Train classifier

            if classif_type == 'knn':
                knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric=get_dist_func(knn_metric))
                knn.fit(visual_words, train_labels)
            elif classif_type == 'svm':
                pass
                #TODO
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

            visual_words_test = np.zeros((len(val_filenames), codebook_size), dtype=np.float32)
            for i in tqdm(range(len(val_filenames))):
                filename = val_filenames[i]
                ima = cv2.imread(filename)
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                kpt = get_keypoints(gray, kp_detector)
                kpt, des = get_descriptors(gray, kpt, desc_type)
                words = codebook.predict(des)
                visual_words_test[i, :] = np.bincount(words, minlength=codebook_size)
                if(normalize):
                    visual_words_test = transformer.transform(visual_words_test)

            # ASSESSMENT OF THE CLASSIFIER
            accuracy = 100 * knn.score(visual_words_test, val_labels)
            accumulated_accuracy.append(accuracy)
        print("codebook size: ", codebook_size, " accuracy: ", np.sum(accumulated_accuracy)/5)
            # Show Confusion Matrix
            # showConfusionMatrix(dist_name_list, conf_mat_list, labels_names)
