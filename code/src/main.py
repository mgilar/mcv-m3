import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm 
from assessment import showConfusionMatrix
from descriptors import get_keypoints, get_descriptors, get_bag_of_words
from classifiers import get_dist_func

#Train parameters
kp_detector = 'sift' #sift 
desc_type = 'sift' #pyramid
codebook_size = 128
classif_type  =  'knn' #svm
knn_metric = 'euclidean'
save_trainData = True


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


def save_data(object, filename):
    filehandler = open(filename, 'w')
    pickle.dump(object, filehandler)


def load_data(filename):
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)

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

    for filename, labels in zip(tqdm(train_images_filenames), train_labels):
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        # 1. Detect keypoints (sift or dense)
        kpt = get_keypoints(gray, kp_detector)
        # 2. Get descriptors (normal sift or spatial pyramid)
        kpt, des = get_descriptors(gray, kpt, desc_type)
        train_desc_list.append(des)
        Train_label_per_descriptor.append(labels)

    D = np.vstack(train_desc_list)

    # 3. Normalize descriptors TODO

    # 4. Create codebook and fit with train dataset
    codebook, visual_words = get_bag_of_words(D, train_desc_list, codebook_size)

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
        codebook = load_data("codebook.pkl")
        visual_words = load_data("visual_words.pkl")

    # VALIDATE CLASSIFIER WITH CROSS-VALIDATION DATASET

    visual_words_test = np.zeros((len(test_images_filenames), codebook_size), dtype=np.float32)
    for i in tqdm(range(len(test_images_filenames))):
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
    showConfusionMatrix(dist_name_list, conf_mat_list, labels_names)
