import random
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import copy
from scipy import signal
from scipy import stats


# from mne.decoding import UnsupervisedSpatialFilter
# from sklearn.decomposition import PCA, FastICA
import seaborn as sns

# from intervals_mat import *

from datetime import datetime
from correlation import (
    correlate_function_2,
    correlate_function,
    correlate_function_right,
    split_list_by_lengths,
    top_indices,
)


from builtins import range


# from evaluate_two import transform_data_not_transpose, print_value_counts
# import evaluate_two
from utils import *

from correlation import split_list_by_lengths

correlation_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

f = open("results.txt", "a")
path_folder = "../500ms/"
file_name = [f for f in os.listdir("../500ms/")]
file_paths = []
for file_number in range(len(file_name)):
    file = os.path.join(path_folder, file_name[file_number])
    file_paths.append(file)

x_samp, y_samp = transform_data_not_transpose(file_name, file_paths, 16)


# print(x_samp.shape)
# x_samp_t = np.transpose(x_samp)
x_samp_tt = np.transpose(x_samp, (0, 2, 1))
print(x_samp_tt.shape)

labels_correlation_windowless = print_value_counts(y_samp)
lengths = []
for kk in labels_correlation_windowless.values():
    lengths.append(kk)
print(lengths)
# Split the data_list based on split_indices
data_merge = split_list_by_lengths(x_samp_tt, lengths)
new_reduced_labels = []

# plt.plot(data_merge[0][0][0])
# plt.show()

# Generate a random number between the intervals
for j in range(len(data_merge)):
    for i in range(len(data_merge[j])):
        # for k in range(len(data_merge[j][i])):
        data_merge[j][i] = stats.zscore(data_merge[j][i])

# plt.plot(data_merge[0][0][0])
# plt.show()


for i in range(len(data_merge)):
    # splice_index = int(len(data_merge[i]) * 0.50)
    random_number = random.randint(0, lengths[i] - 49)
    data_merge[i] = data_merge[i][random_number : random_number + 48]
    new_reduced_labels.extend([i] * 48)


data_merge_I = []
data_merge_II = []

for i in range(len(data_merge)):
    data_merge_I.append(data_merge[i][0:24])
    data_merge_II.append(data_merge[i][24:48])
# x_samp_tt = np.concatenate(data_merge)
# print("ciao")
# print(x_samp_tt.shape)
new_reduced_labels = np.array(new_reduced_labels)

data_merge_conc = np.concatenate(data_merge, axis=0)
# print(data_merge_conc.shape)
counter = 0

epsilon = 0.8
# for k in range(16):
#     for i in range(len(data_merge)):
#         score = []
#         for j in range(len(data_merge_I[0])):
#             score.append(
#                 max(
#                     signal.correlate(
#                         data_merge_I[i][j][k, 0:500],
#                         data_merge_II[i][j][k, 0:500],
#                         mode="same",
#                     )
#                 )
#             )
#         # print("\n")
#         # print(score)
#         # print(len(score))
#         # print("\n")
#         correlation_scores[k] += 0.8 * np.median(score)

# # print(correlation_scores)

# for k in range(16):
#     for i in range(len(data_merge)):
#         score = []
#         for j in range(len(data_merge_I[0])):
#             x = copy.copy(data_merge_II)
#             x.pop(i)
#             if j < 8:
#                 score.append(
#                     max(
#                         signal.correlate(
#                             data_merge_I[i][j][k, 0:500],
#                             x[0][j][k, 0:500],
#                             mode="same",
#                         )
#                     )
#                 )
#             elif j > 7 and j < 16:
#                 score.append(
#                     max(
#                         signal.correlate(
#                             data_merge_I[i][j][k, 0:500],
#                             x[1][j][k, 0:500],
#                             mode="same",
#                         )
#                     )
#                 )
#             else:
#                 score.append(
#                     max(
#                         signal.correlate(
#                             data_merge_I[i][j][k, 0:500],
#                             x[2][j][k, 0:500],
#                             mode="same",
#                         )
#                     )
#                 )
#         # print("\n")
#         # print(score)
#         # print(len(score))
#         # print("\n")
#         correlation_scores[k] -= 0.2 * np.median(score)

for i in range(100):
    print("cycling...")
    for k in range(16):
        same_class_score = []
        diff_class_score = []
        for ch2 in range(16):
            if k != ch2:
                for i in range(len(data_merge)):
                    for h in range(len(data_merge)):
                        if i == h:
                            for j in range(5):
                                same_class_score.append(
                                    max(
                                        signal.correlate(
                                            data_merge_I[i][k][j, 0:2500],
                                            data_merge_II[h][ch2][j, 0:500],
                                            mode="same",
                                        )
                                    )
                                )
                        else:
                            for j in range(5):
                                diff_class_score.append(
                                    max(
                                        signal.correlate(
                                            data_merge_I[i][k][j, 0:2500],
                                            data_merge_II[h][ch2][j, 0:500],
                                            mode="same",
                                        )
                                    )
                                )
        # print(len(same_class_score))
        # print(len(diff_class_score))
        correlation_scores[k] += epsilon * np.median(same_class_score) + (
            1 - epsilon
        ) * -1 * np.median(diff_class_score)


print(correlation_scores)

print(top_indices(correlation_scores, 16))

# correlate_function_right(
#     16,
#     4,
#     x_train,
#     y_train,
#     labels_correlation_windowless,
#     f,
#     500,
#     correlation_scores,
#     # voting,
# )

# for i, j in enumerate(correlation_scores):
#     correlation_scores[i] = correlation_scores[i] / 5
# print(correlation_scores)
