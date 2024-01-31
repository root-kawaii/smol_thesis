from cmath import log
import os
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
import mne
from CSP1 import csp2
from neo import io
from mne.decoding import CSP
import math
from scipy import signal
from scipy import stats
import pycwt
import asrpy
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cbook as cbook

files = [f for f in os.listdir("../pso/data")]
# files_nev = [f for f in os.listdir("../pso/data_nev")]
data_list = []
labels_list = []
slack_var = 0
var_data = []
mean_data = []
list_mass = []
entropy_data = []
corr_list = []
big_x = []
epsilon = 0.9
list_of_raw = []


f = open("results.txt", "w")


# for count, file in enumerate(files):
#     r = io.BlackrockIO(
#         filename="../pso/data/" + file)
#     bl = r.read_block(lazy=False)
#     # bl = r.get_data().T
#     # print(len(bl[0]))
#     seg = bl.segments[0]
#     # two signals, we use first with 16 channels
#     x = seg.analogsignals[0].transpose()
#     y = seg.analogsignals[1]
#     for i in range(16):
# probability for each voltage value dictionary, needed to calculate entropy
# mass_prob_dic = {}
# # initialize list for each channel
# mean_data.append(0)
# var_data.append(0)
# entropy_data.append(0)
# for j in x[i]:
#     # update dictionary probabilities
#     if str(j) not in mass_prob_dic.keys():
#         mass_prob_dic[str(j)] = 1/len(x[i])
#     else:
#         mass_prob_dic[str(j)] += 1/len(x[i])
#     # sum and at end of loop divide by length
#     if (mean_data[i] == 0):
#         mean_data[i] = j
#     else:
#         mean_data[i] += j
# # finish calculation of mean
# mean_data[i] = mean_data[i]/len(x[i])
# # print(mass_prob_dic)
# list_mass.append(mass_prob_dic)
# # perform caluculation for entropy and variance
# # formulas from paper // Channel selection for automatic seizure detection
# # https://www.sciencedirect.com/science/article/pii/S1388245711003774?via=ihub
# for j, k in enumerate(x[i]):
#     if (entropy_data[i] == 0):
#         entropy_data[i] = mass_prob_dic[str(
#             k)] * log(mass_prob_dic[str(k)], 2)
#     else:
#         entropy_data[i] += mass_prob_dic[str(k)] * \
#             log(mass_prob_dic[str(k)], 2)
#     if (var_data[i] == 0):
#         var_data[i] = pow(x[i, j]-mean_data[i], 2)
#     else:
#         var_data[i] += pow(x[i, j]-mean_data[i], 2)
# # fix sign of entropy
# entropy_data[i] = -1 * entropy_data[i]
# # divide variance
# var_data[i] = var_data[i] / len(x[i])
# f.write("Variance channel  " + str(i) + "  " + str(var_data[i]) + "\n")
# f.write("Entropy channel  " + str(i) +
#         "  " + str(entropy_data[i]) + "\n")


# for count, file in enumerate(files):
#     num_electrodes = 0
#     num_sensors = 0
#     nsx_filepath = file
#     # r = mne.io.read_raw_nsx("../pso/data/" + file, preload=True)
#     r = io.BlackrockIO(
#         filename="../pso/data/" + file)
#     bl = r.read_block(lazy=False)
#     # bl = r.get_data().T
#     # print(len(bl[0]))
#     seg = bl.segments[0]
#     x = seg.analogsignals[0].transpose()
#     # y = seg.analogsignals[1].transpose()

#     # n_chan, n_pnts = len(x), len(x[0])
#     data = np.zeros((16, 1000000), dtype=float)
#     for i in range(16):
#         # if position[i] == 1:
#         data[i, 0:1000000] = x[i, 0:1000000]

#     data_list.append(data)
#     labels_list.extend([count] * 1000000)

# data_merge = np.concatenate(data_list, axis=1)
# print(data_merge.shape)

# for i in range(16):
#     # Cross-Correlation of channels
#     same_class = 0
#     diff_class = 0
#     data_merge[i] = stats.zscore(data_merge[i])
#     listona = []
#     for k in range(16):
#         if (i != k):
#             new = signal.correlate(
#                 data_merge[i, 0:1000000], data_merge[k, 0:1000000], mode='same')
#             new4 = signal.correlate(
#                 data_merge[i, 1000000:], data_merge[k, 1000000:], mode='same')
#             new = max(new+new4)
#             new2 = signal.correlate(
#                 data_merge[i, 0:1000000], data_merge[k, 1000000:], mode='same')
#             new3 = signal.correlate(
#                 data_merge[i, 1000000:], data_merge[k, 0:1000000], mode='same')
#             new2 = max(new2 + new3)

#             if (same_class < new):
#                 same_class = new
#             if (diff_class < new2):
#                 diff_class = new2
#             # if (c_2_varr < new3):
#             #     c_2_varr = new3
#             # if (c_2_other_varr < new4):
#             #     c_2_other_varr = new4
#             print("\n")
#             print("\n")
#             listona.append(new)
#             listona.append(new2)
#     a = 0
#     b = 0
#     for j in range(0, 16, 2):
#         a += listona[j]
#         b += listona[j+1]
#     a = a / 16
#     b = -b / 16
#     metric = epsilon*a + (1-epsilon)*b
#     corr_list.append(metric)

# for i in corr_list:
#     f.write("Score  " + str(i) + "\n")
# f.write("Now sorted...  " + "\n")
# corr_list.sort()
# for i in corr_list:
#     f.write("Score  " + str(i) + "\n")
# f.close()


for count, file in enumerate(files):
    num_electrodes = 0
    num_sensors = 0
    nsx_filepath = file
    raw = mne.io.read_raw_nsx("../pso/data/" + file, preload=True)
    # raw.plot()
    # faster but more of a pain
    # r = io.BlackrockIO(
    #     filename="../pso/data/" + file)
    # bl = r.read_block(lazy=False)
    # raw = r.get_data()
    # print(len(bl[0]))
    # seg = bl.segments[0]
    # x = seg.analogsignals[0].transpose()
    # print(raw.shape)
    # y = seg.analogsignals[1].transpose()
    # Set montage
    # montage = mne.channels.make_standard_montage('easycap-M1')
    # raw.set_montage(montage, verbose=False)

    # downsample for faster computation
    raw.resample(512)

    # apply a highpass filter from 1 Hz upwards
    # replace baselining with high-pass
    # raw.filter(1., None, fir_design='firwin')

    # Apply the ASR
    print(raw.get_data())
    # asr = asrpy.ASR(sfreq=30000, cutoff=15)
    # asr.fit(raw)
    # raw = asr.transform(raw)
    print(raw.get_data())

    # Create an average using the cleaned data
    # clean_avg = mne.Epochs(raw, -0.1, 1.5, proj=False,
    #                        picks=None, baseline=None, preload=True,
    #                        verbose=False).average()
    x = raw.get_data()
    data = np.zeros((16, 100000), dtype=float)
    for i in range(16):
        # if position[i] == 1:
        data[i, 0:100000] = x[i, 0:100000]

       # data_list.append(data)
        labels_list.extend([count] * 100000)
        # ch_names = [str(i) for i in range(16)]
        # info = mne.create_info(ch_names=ch_names, sfreq=int(
        #     seg.analogsignals[0].sampling_rate.magnitude))
        # raw = mne.io.RawArray(x, info)
        # sfreq = raw.info["sfreq"]
        # asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=20)
        # asr.fit(raw)
        # raw = asr.transform(raw)
        # print(raw.shape)
        # M, T = asrpy.asr_calibrate(x, sfreq, cutoff=15)
        # # apply it
        # clean_array = asrpy.asr_process(x, sfreq, M, T)
        fig, axd = plt.subplot_mosaic(
            [["image", "density"],
             ["EEG", "EEG"]],
            layout="constrained",
            # "image" will contain a square image. We fine-tune the width so that
            # there is no excess horizontal or vertical margin around the image.
            width_ratios=[1.05, 2],
        )

        plt.plot(data[i])
        plt.show()
        data_list.append(raw[0])

# n_chan, n_pnts = len(x), len(x[0])
# data = np.zeros((16, 100000), dtype=float)
# for i in range(16):
#     # if position[i] == 1:
#     data[i, 0:100000] = x[i, 0:100000]

#    # data_list.append(data)
#     labels_list.extend([count] * 100000)
#     ch_names = [str(i) for i in range(16)]
#     info = mne.create_info(ch_names=ch_names, sfreq=int(
#         seg.analogsignals[0].sampling_rate.magnitude))

# raw = mne.io.RawArray(x, info)
# raw.filter(1., None, fir_design='firwin')
# events, _ = mne.events_from_annotations(raw, verbose=False)
# sfreq = raw.info["sfreq"]
# print(events)
# # asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=20)
# # asr.fit(raw)
# # raw = asr.transform(raw)
# # print(raw.shape)
# M, T = asrpy.asr_calibrate(x, sfreq, cutoff=15)
# # apply it
# clean_array = asrpy.asr_process(x, sfreq, M, T)
# data_list.append(raw[0])


# data_merge = np.concatenate(data_list, axis=1)
# print('ciao')
# res = pycwt.xwt(data_merge[0], data_merge[1], 30000)
# one = pycwt.cwt(data_merge[0], 30000)
# two = pycwt.cwt(data_merge[1], 30000)
# print(two[0].shape)
# three = np.matmul(one[0], two[0])
# four = np.divide(res, three)
