import os
from matplotlib import pyplot as plt
import numpy as np
import sklearn

# from sklearn.base import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import *
import mne
from CSP1 import csp2
from neo import io
from mne.decoding import CSP
import asrpy
from asrpy import asr_calibrate, asr_process, clean_windows
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
import tensorflow as tf
import seaborn as sns
from modelz import *
import random
from scipy import stats
from scipy import signal

from datetime import datetime


def split_list_by_lengths(data_list, lengths):
    result = []
    start = 0
    for length in lengths:
        result.append(data_list[:, start : start + length])
        start += length
    return result


def correlate_function(
    num_features, classes, data_merge, labels_correlation, f, window
):
    print(data_merge.shape)
    epsilon = 0.9
    ulter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lengths = []
    for kk in labels_correlation.values():
        lengths.append(kk)
    print(lengths)
    # Split the data_list based on split_indices
    data_merge = split_list_by_lengths(data_merge, lengths)
    print(len(data_merge[0]))
    # result = np.array(result)
    # print(len(result[1][2]))
    for o in range(len(data_merge)):
        for i in range(num_features):
            data_merge[o][i] = stats.zscore(data_merge[o][i])
    for n in range(classes):
        corr_list = []
        for i in range(num_features):
            # Cross-Correlation of channels
            same_col = 0
            diff_col = 0
            for k in range(num_features):
                if i != k:
                    for j in range(classes):
                        # print(data_merge[n][i])
                        # print(len(data_merge[n][i]))
                        new = signal.correlate(
                            data_merge[n][i], data_merge[j][k], mode="same"
                        )
                        # print(new)
                        new = max(new)
                        if n == j:
                            same_col += new
                        else:
                            diff_col += new

            avg_same_col = same_col / num_features - 1
            avg_diff_col = diff_col / num_features * 3
            metric = epsilon * avg_same_col + (1 - epsilon) * avg_diff_col
            corr_list.append(metric)

        for i in corr_list:
            f.write("Score  " + str(i) + "\n")
        f.write("Now sorted...  " + "\n")
        # corr_list.sort()
        print(corr_list)
        for j, i in enumerate(corr_list):
            ulter[j] += i
            # print("U " + str(ulter[j]))
    for h in ulter:
        f.write("B " + str(h) + "\n")
        print("U " + str(h))
