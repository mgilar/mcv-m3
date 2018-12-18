import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

from assessment import showConfusionMatrix
from descriptors import get_keypoints, get_descriptors, get_bag_of_words
from classifiers import get_dist_func

#Train parameters
kp_detector = 'dense' #sift
desc_type = 'sift' #pyramid
codebook_size = 128
classifier  =  'knn' #svm
knn_metric = 'euclidean'

def print_res(names, accuracies, times):
    """
    Function used when showing results
    """
    def get_ristra_string(l, find_highest=False):
        string = ""
        for v in l:
            if(isinstance(v, float)):
                substring = "{0:.2f}".format(v)[:4]
                if(find_highest and v == max([x for x in l if isinstance(x, float)])):
                    substring = substring+"*"
            else:
                substring=v[:4]
            string+=substring+"\t"
        return string+"\n"
    s  = "     \t"+get_ristra_string(names)
    s += " Acc:\t"+get_ristra_string(accuracies, True)
    s += "Time:\t"+get_ristra_string(times, False)
    print(s)


if __name__ == '__main__':

    #Load train and test files
    train_images_filenames = pickle.load(open('./train_images_filenames.dat', 'rb'))
    test_images_filenames = pickle.load(open('./test_images_filenames.dat', 'rb'))
    train_labels = pickle.load(open('./train_labels.dat', 'rb'))
    test_labels = pickle.load(open('./test_labels.dat', 'rb'))

    #Split train dataset for cross-validation
    #TODO

    # TRAIN CLASSIFIER

    train_desc_list = []
    Train_label_per_descriptor = []

    for filename, labels in zip(train_images_filenames, train_labels):
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        # 1. Detect keypoints (sift or dense)
        kpt = get_keypoints(gray, kp_detector)
        # 2. Get descriptors (normal sift or spatial pyramid)
        des = get_descriptors(gray, kpt, desc_type)
        train_desc_list.append(des)
        Train_label_per_descriptor.append(labels)

    D = np.vstack(train_desc_list)

    # 3. Normalize descriptors TODO

    # 4. Create codebook and fit with train dataset
    codebook, visual_words = get_bag_of_words(D, train_desc_list, codebook_size)

    # 5. Train classifier

    if classifier == 'knn':
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric=get_dist_func(knn_metric))
        knn.fit(visual_words, train_labels)
    elif classifier == 'svm':
        pass
        #TODO
    else:
        raise (NotImplemented("classifier not implemented or not recognized:" + str(classifier)))

    # 6. Save data in pickle TODO


    # VALIDATE CLASSIFIER WITH CROSS-VALIDATION DATASET

    visual_words_test = np.zeros((len(test_images_filenames), codebook_size), dtype=np.float32)
    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt = get_keypoints(gray, kp_detector)
        des = get_descriptors(gray, kpt, desc_type)
        words = codebook.predict(des)
        visual_words_test[i, :] = np.bincount(words, minlength=codebook_size)

    # ASSESSMENT OF THE CLASSIFIER
    accuracy = 100 * knn.score(visual_words_test, test_labels)
    print(accuracy)
    # Show Confusion Matrix
    #showConfusionMatrix(dist_name_list, conf_mat_list, labels_names)
