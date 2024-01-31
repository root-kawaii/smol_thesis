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


def correlate_function(num_features, num_files, data_merge, f):
    epsilon = 0.9
    ulter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for o in range(10):
        data_merge[o] = stats.zscore(data_merge[o])
    for n in range(10):
        corr_list = []
        for i in range(num_features):
            # Cross-Correlation of channels
            same_class = 0
            diff_class = 0

            listona = []
            same_col = 0
            diff_col = 0
            for k in range(num_features):
                if (i != k):
                    for j in range(10):
                        # print(data_merge[n][i])
                        # print(len(data_merge[n][i]))
                        new = signal.correlate(
                            data_merge[n][i], data_merge[j][k], mode='same')
                        # print(new)
                        new = max(new)
                        if (n == j):
                            same_col += new
                        else:
                            diff_col += new

            avg_same_col = same_col / num_features-1
            avg_diff_col = diff_col / num_features * num_files
            metric = epsilon*avg_same_col + (1-epsilon)*avg_diff_col
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
        f.write(str(h) + "\n")
        print("U " + str(h))
