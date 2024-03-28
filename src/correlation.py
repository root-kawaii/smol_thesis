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
import seaborn as sns
from modelz import *
import random
from scipy import stats
from scipy import signal

from datetime import datetime
from builtins import range


def split_list_by_lengths(data_list, lengths):
    result = []
    start = 0
    for length in lengths:
        result.append(data_list[start : start + length])
        start += length
    return result


def max_values_and_indices(arr, ind):
    # Create a list of tuples with values and their indices
    indexed_arr = list(enumerate(arr))
    # Sort the list based on values in descending order
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=False)
    # Take the top 5 values and their indices
    top_5_values = [x[1] for x in sorted_arr[-ind:]]
    top_5_indices = [x[0] for x in sorted_arr[-ind:]]
    return top_5_values, top_5_indices


def correlate_function(
    num_features, classes, data_merge, labels_correlation, f, window, correlation_scores
):

    epsilon = 0.8
    lengths = []
    for kk in labels_correlation.values():
        lengths.append(kk)
    print(lengths)
    # Split the data_list based on split_indices
    data_merge = split_list_by_lengths(data_merge, lengths)
    # result = np.array(result)
    # print(len(result[1][2]))
    corr_list = []

    for o in range(len(data_merge)):
        for i in range(num_features):
            data_merge[o][i] = stats.zscore(data_merge[o][i])
    for i in range(num_features):
        print("...wip...")

        same_col = 0
        diff_col = 0
        for n in range(classes):
            # Cross-Correlation of channels
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
        metric = (epsilon * (+1) * avg_same_col) + ((1 - epsilon) * (-1) * avg_diff_col)
        corr_list.append(metric)

    for j, i in enumerate(corr_list):
        correlation_scores[j] += i
        # print("U " + str(ulter[j]))
    for h in correlation_scores:
        # f.write("B " + str(h) + "\n")
        print("U " + str(h))


def correlate_function_2(
    num_features,
    classes,
    data_merge,
    labels_correlation,
    f,
    window,
    correlation_scores,
    voting,
):

    epsilon = 0.8
    lengths = []
    for kk in labels_correlation.values():
        lengths.append(kk)
    print(lengths)
    # Split the data_list based on split_indices
    data_merge = split_list_by_lengths(data_merge, lengths)
    print(len(data_merge))
    # print(len(data_merge[0]))
    # result = np.array(result)
    # print(len(result[1][2]))
    corr_list = []

    overall_list = np.zeros((16, 16))
    for o in range(len(data_merge)):
        for i in range(num_features):
            data_merge[o][i] = stats.zscore(data_merge[o][i])
    for i in range(num_features):
        print("...wip...")

        same_col = []
        for n in range(classes):
            # Cross-Correlation of channels
            for k in range(num_features):
                if i != k:
                    # print(data_merge[n][i])
                    # print(len(data_merge[n][i]))
                    new = signal.correlate(
                        data_merge[n][i], data_merge[n][k], mode="same"
                    )
                    # print(new)
                    new = max(new)
                    overall_list[i, k] += new
                else:
                    overall_list[i, k] += 0
    print(overall_list)
    for i in range(num_features):
        val, index = max_values_and_indices(overall_list[i], 8)
        for va in index:
            print(va)
            voting[va] += 1


def correlate_function_right(
    num_features,
    classes,
    data_merge,
    labels,
    labels_correlation,
    f,
    window,
    correlation_scores,
):

    epsilon = 0.8
    lengths = []
    # need to re-write z_score with new correlation arguments
    for i in range(num_features):
        for o in data_merge:
            o[i] = stats.zscore(o[i])
    for i in range(num_features):
        current_time = datetime.now()
        print(str(current_time))

        # same_col = 0
        # diff_col = 0
        same_col = []
        diff_col = []
        for j, k in enumerate(data_merge):
            first_class = labels[j]
            same_col_counter = 0
            diff_col_counter = 0
            for q, w in enumerate(data_merge):
                second_class = labels[q]
                if (k != w).all():
                    # print(data_merge[n][i])
                    # print(len(data_merge[n][i]))
                    new = signal.correlate(k[i], w[i], mode="full")
                    new = max(new)
                    if first_class == second_class:
                        # same_col += new
                        same_col.append(new)
                        same_col_counter += 1
                    else:
                        # diff_col += new
                        diff_col.append(new)
                        diff_col_counter += 1

            # avg_same_col = same_col / same_col_counter
            # avg_diff_col = diff_col / diff_col_counter
            avg_same_col = np.median(same_col)
            avg_diff_col = np.median(diff_col)
            metric = (epsilon * (+1) * avg_same_col) + (
                (1 - epsilon) * (-1) * avg_diff_col
            )
            correlation_scores[i] += metric
        print(correlation_scores[i])
