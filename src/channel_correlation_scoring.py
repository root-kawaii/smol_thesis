import neo
import os
from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import copy
import scipy.io

# from sklearn.svm import SVC
import mne

# from mne.decoding import CSP
import asrpy
from asrpy import asr_calibrate, asr_process, clean_windows

# from mne.decoding import UnsupervisedSpatialFilter
# from sklearn.decomposition import PCA, FastICA
import tensorflow as tf
import seaborn as sns
from modelz import *

# from intervals_mat import *

from datetime import datetime
from correlation import (
    correlate_function_2,
    correlate_function,
    correlate_function_right,
    split_list_by_lengths,
)
import pywt

from builtins import range


tfk = tf.keras
tfkl = tf.keras.layers

# from evaluate_two import transform_data_not_transpose, print_value_counts
# import evaluate_two
from utils import *

from correlation import split_list_by_lengths

correlation_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

f = open("results.txt", "a")
path_folder = "../500ms/"
file_name = [f for f in os.listdir("../500ms/")]
file_paths = []
for file_number in range(len(file_name)):
    file = os.path.join(path_folder, file_name[file_number])
    file_paths.append(file)

x_samp, y_samp = transform_data_not_transpose(file_name, file_paths, 16)

print(x_samp.shape)
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
for i in range(len(data_merge)):
    splice_index = int(len(data_merge[i]) * 0.50)
    data_merge[i] = data_merge[i][0:47]
    new_reduced_labels.extend([i] * 47)

data_merge_I = []
data_merge_II = []

for i in range(len(data_merge)):
    data_merge_I.append(data_merge[i][0:23])
    data_merge_II.append(data_merge[i][23:47])
# x_samp_tt = np.concatenate(data_merge)
# print("ciao")
# print(x_samp_tt.shape)
new_reduced_labels = np.array(new_reduced_labels)

for i in range(len(data_merge)):
    for j in range(len(data_merge[0])):
        """from data_merge_II remove row each iteration according to the one we are iterating on in data_merge_I"""
        x = copy.copy(data_merge_II)
        x.pop(i)
        print(len(data_merge_II))
        if j < 8:
            do(data_merge_I[i][j], x[0][j])
        elif j > 7 and j < 16:
            do(data_merge_I[i][j], x[1][j])
        else:
            do(data_merge_I[i][j], x[2][j])

correlate_function_right(
    16,
    4,
    x_train,
    y_train,
    labels_correlation_windowless,
    f,
    500,
    correlation_scores,
    # voting,
)

for i, j in enumerate(correlation_scores):
    correlation_scores[i] = correlation_scores[i] / 5
print(correlation_scores)
