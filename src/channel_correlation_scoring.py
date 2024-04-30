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
correlation_scores_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
correlation_scores_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

f = open("results.txt", "a")
path_folder = "../data/500ms/"
file_name = [f for f in os.listdir("../data/500ms/")]
file_paths = []
for file_number in range(len(file_name)):
    file = os.path.join(path_folder, file_name[file_number])
    file_paths.append(file)

x_samp, y_samp = transform_data_not_transpose(file_name, file_paths, 16)


print(x_samp.shape)
# x_samp_t = np.transpose(x_samp)
x_samp_tt = np.transpose(x_samp, (0, 2, 1))
print(x_samp_tt.shape)


################### generate substitute ################################


for i in range(len(x_samp_tt)):
    x_samp_tt[i] = stats.zscore(x_samp_tt[i])

sub = np.zeros(2500)
summ = []
for k in range(15):
    for i in range(len(x_samp_tt)):
        for j in range(len(x_samp_tt[i, 0])):
            # print(j)
            summ.append(x_samp_tt[i, k, j])


print(len(summ))
# Calculate average (mean)
average = np.mean(summ)
print(average)
# Calculate standard deviation
std_dev = np.std(summ)
print(std_dev)


# Parameters for the signal
signal_mean = average  # Mean of the signal
signal_std = std_dev  # Standard deviation of the signal
num_data_points = 2500  # Number of data points

# Generate the signal without noise
for j in range(15):
    for i in range(2013):
        signalo = np.random.normal(signal_mean, signal_std, num_data_points)
        x_samp_tt[i, j] = signalo


# Plot the noisy signal
plt.figure(figsize=(10, 6))
plt.plot(signalo, color="b", label="Noisy Signal")
plt.xlabel("Data Points")
plt.ylabel("Amplitude")
plt.title("Noisy Signal with AWGN")
plt.legend()
plt.grid(True)
plt.show()
plt.plot(x_samp_tt[0, 0], color="b", label="Noisy Signal")
plt.show()


#############################################################################


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
# for j in range(len(data_merge)):
#     for i in range(len(data_merge[j])):
#         # for k in range(len(data_merge[j][i])):
#         data_merge[j][i] = stats.zscore(data_merge[j][i])

# plt.plot(data_merge[0][0][0])
# plt.show()


for i in range(len(data_merge)):
    # splice_index = int(len(data_merge[i]) * 0.50)
    random_number = random.randint(0, lengths[i] - 81)
    data_merge[i] = data_merge[i][random_number : random_number + 80]
    new_reduced_labels.extend([i] * 80)


data_merge_I = []
data_merge_II = []

for i in range(len(data_merge)):
    data_merge_I.append(data_merge[i][0:40])
    data_merge_II.append(data_merge[i][40:80])


# x_samp_tt = np.concatenate(data_merge)
# print("ciao")
# print(x_samp_tt.shape)
new_reduced_labels = np.array(new_reduced_labels)

data_merge_conc = np.concatenate(data_merge, axis=0)
# print(data_merge_conc.shape)
counter = 0

epsilon = 0.8


print(correlation_scores)
# for i in range(10):
#     for batch in range(2):
#         for k in range(8):
#             k = k + (8 * batch)
#             for i in range(len(data_merge)):
#                 score = []
#                 for j in range(len(data_merge_I[0])):
#                     score.append(
#                         max(
#                             signal.correlate(
#                                 data_merge_I[i][j][k, 0:2500],
#                                 data_merge_II[i][j][k, 0:500],
#                                 mode="same",
#                             )
#                         )
#                     )
#                 # print("\n")
#                 # print(score)
#                 # print(len(score))
#                 # print("\n")
#                 correlation_scores_1[k] += np.median(score)

#         for k in range(8):
#             k = k + (8 * batch)
#             for i in range(len(data_merge)):
#                 score = []
#                 for j in range(len(data_merge_I[0])):
#                     x = copy.copy(data_merge_II)
#                     x.pop(i)
#                     if j < 8:
#                         score.append(
#                             max(
#                                 signal.correlate(
#                                     data_merge_I[i][j][k, 0:2500],
#                                     x[0][j][k, 0:500],
#                                     mode="same",
#                                 )
#                             )
#                         )
#                     elif j > 7 and j < 16:
#                         score.append(
#                             max(
#                                 signal.correlate(
#                                     data_merge_I[i][j][k, 0:2500],
#                                     x[1][j][k, 0:500],
#                                     mode="same",
#                                 )
#                             )
#                         )
#                     else:
#                         score.append(
#                             max(
#                                 signal.correlate(
#                                     data_merge_I[i][j][k, 0:2500],
#                                     x[2][j][k, 0:500],
#                                     mode="same",
#                                 )
#                             )
#                         )
#                 # print("\n")
#                 # print(score)
#                 # print(len(score))
#                 # print("\n")
#                 correlation_scores_2[k] -= np.median(score)

# for i in range(16):
#     correlation_scores[i] += (
#         epsilon * correlation_scores_1[i] / 100
#         + (1 - epsilon) * correlation_scores_2[i] / 100
#     )


for y in range(20):
    print("cycling...")
    # for batch in range(2):
    for k in range(16):
        # k = k + (8 * batch)
        same_class_score = []
        diff_class_score = []
        # for batch2 in range(2):
        for ch2 in range(16):
            # ch2 = ch2 + (8 * batch)
            if k != ch2:
                for i in range(len(data_merge)):
                    for h in range(len(data_merge)):
                        if i == h:
                            for j in range(39):
                                same_class_score.append(
                                    max(
                                        signal.correlate(
                                            data_merge_I[i][j][k, 0:2500],
                                            data_merge_II[h][j][ch2, 0:500],
                                            mode="same",
                                        )
                                    )
                                )
                        else:
                            for j in range(39):
                                diff_class_score.append(
                                    max(
                                        signal.correlate(
                                            data_merge_I[i][j][k, 0:2500],
                                            data_merge_II[h][j][ch2, 0:500],
                                            mode="same",
                                        )
                                    )
                                )
        # print(len(same_class_score))
        # print(len(diff_class_score))
        correlation_scores_1[k] += np.median(same_class_score)
        correlation_scores_2[k] += np.median(diff_class_score)
        correlation_scores[k] += epsilon * np.median(same_class_score) + (
            1 - epsilon
        ) * -1 * np.median(diff_class_score)

print(correlation_scores_1)
print(correlation_scores_2)
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


# x = [
#     27741.894359805046,
#     30456.98012862733,
#     28359.515126905902,
#     27196.5705345903,
#     31359.662968022265,
#     33643.65796149252,
#     25005.298644436156,
#     26715.147127287873,
#     31610.14788118171,
#     30488.104059332054,
#     25407.14354390039,
#     26306.929121814388,
#     31785.600190838904,
#     31555.79461742658,
#     28593.309649886676,
#     28904.723617399362,
# ]

# x_2 = [
#     732.482053962357,
#     761.6008655195773,
#     737.1869873889632,
#     720.8385499325378,
#     770.6208815696847,
#     795.8753604830462,
#     692.7994325041677,
#     722.4649055946104,
#     769.7374985833649,
#     768.6664703563209,
#     703.9201442489036,
#     711.7144499125869,
#     770.9491007645128,
#     774.3017718104165,
#     734.7361788866406,
#     741.8855067047688,
# ]

# y = [
#     -27331.501740066582,
#     -29629.078540779974,
#     -28416.996871003208,
#     -26825.654251823562,
#     -30908.243196558695,
#     -33515.918534939556,
#     -24834.01815400629,
#     -26993.645962758335,
#     -30985.941116820486,
#     -30964.55140997413,
#     -25842.201600594155,
#     -26473.052317600082,
#     -31325.47416671485,
#     -31556.608934585965,
#     -27200.10897146471,
#     -28918.706124248067,
# ]

# y_2 = [
#     729.0314921657954,
#     758.4481368845007,
#     735.0648749393167,
#     722.4477020334097,
#     774.9469232883007,
#     795.413411345846,
#     691.5351060644781,
#     722.8723226647076,
#     770.8965811161656,
#     766.5515475254643,
#     701.7109935117617,
#     715.1300164176663,
#     772.8594611033423,
#     774.0988579742669,
#     733.0037817330648,
#     743.1391090211035,
# ]

# x = correlation_scores_1
# y = correlation_scores_2
# z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# z_t = []
# w = []

# for i in range(11):
#     epsilon = 0.1 + i / 10
#     z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#     for i in range(16):
#         z[i] += epsilon * x[i] / 8 + (1 - epsilon) * -1 * y[i] / 8
#     z_t.append(z)

#     print(top_indices(z, 16))
#     print(z)

# import csv

# with open("z_list.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(
#         [
#             "Index",
#             "0",
#             "1",
#             "2",
#             "3",
#             "4",
#             "5",
#             "6",
#             "7",
#             "8",
#             "9",
#             "10",
#             "11",
#             "12",
#             "13",
#             "14",
#             "15",
#         ]
#     )
#     # for j in range(len(z)):
#     for i, value in enumerate(z_t):
#         writer.writerow([i / 10] + value)


# print(path_folder)
