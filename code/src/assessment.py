from matplotlib import pyplot as plt
import cv2

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

#TODO plot other assessment metrics

def showConfusionMatrix(dist_name_list,conf_mat_list, labels_names):
    # Show Confusion Matrix
    show_matrix = "bray"
    for i, dist_name in enumerate(dist_name_list):
        if (show_matrix[:4] == dist_name[:4]):
            conf = conf_mat_list[i]
            fig, axs = plt.subplots(1, 1)
            axs.axis('tight')
            axs.axis('off')
            the_table = axs.table(cellText=conf, rowLabels=labels_names, colLabels=labels_names, loc='center')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(12)
            the_table.scale(2, 1)
            fig.tight_layout()
            plt.show()