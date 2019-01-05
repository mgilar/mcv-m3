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

def plot(x, y, label="accuracy", title="Result", xlabel="¿?¿?"):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y, "r.-",label="accuracy", markersize=10.0)
    ax.legend(loc="upper right")
    for i, j in zip(x, y):
        ax.annotate(str(j)[:4], xy=(i,j))
    ax.set_title(title)
    plt.xlabel(xlabel)

def plot3d(x, y, z, label="accuracy", title="Result", xlabel="¿?¿?"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_wireframe(x, y, z, )
    ax.legend(loc="upper right")
    for i, j in zip(x, y):
        ax.annotate(str(j)[:4], xy=(i,j))
    ax.set_title(title)
    plt.xlabel(xlabel)

def get_max_key(d):
    return max(d.keys(), key=lambda x: d[x])


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