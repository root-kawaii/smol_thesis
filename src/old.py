import string
from sklearn import metrics

from sklearn.utils import shuffle
import mne
import neo
import numpy as np

from neo import io

import os

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

files = [f for f in os.listdir("../pso/data")]

data_list = []
raw_list = []
labels = np.chararray((len(files) * 5000))

position = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# def evaluateSol(position):

for count, file in enumerate(files):
    nsx_filepath = file

    r = io.BlackrockIO(filename="../pso/data/" + file)
    bl = r.read_block(lazy=False)
    seg = bl.segments[0]
    # transpose cause channels and inputs were swapped
    # select [0] cause in [1] we find flexsensor pressuresensor motors
    x = seg.analogsignals[0].transpose()
    # number of channels and number of points
    (
        n_chan,
        n_pnts,
    ) = len(
        x
    ), len(x[0])
    # ch_names = list()
    # create empty data with right sizes
    data = np.zeros((n_chan, 5000), dtype=float)
    for i in range(n_chan):
        # if our proposed position has 1 it means the corrisponding feature is enabled,
        #  otherwise we don't want that channel
        if position[i] == 1:
            data[i, 0:5000] = x[i][0:5000]
        else:
            data[i, 0:5000] = None

    sfreq = int(seg.analogsignals[0].sampling_rate.magnitude)
    # kinda useless, i just write the number of electrode to comply with previous code and fill the create_info function
    ch_names = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
    # create array with mne library
    raw = mne.io.RawArray(data, info)
    # print('/n/n')
    # print(raw)
    data_list.append(data)
    raw_list.append(raw)
    s = [count] * 5000
    print(s)
    labels[count * 5000 : ((count + 1) * 5000)] = s
    print(labels)

    # # need to understand other side of dataset
    # y = seg.analogsignals[1].shape
    # print(y)
    # print(seg.analogsignals[1][1])

    # nsx_filepath = "prop_angle_-20_inpoints_0.389_trial_009.ns5"

    # r = io.BlackrockIO(
    #     filename="../pso/prop_angle_-20_inpoints_0.389_trial_009.ns5")
    # bl = r.read_block(lazy=False)
    # seg = bl.segments[0]
    # # transpose cause channels and inputs were swapped
    # # select [0] cause in [1] we find flexsensor pressuresensor motors
    # x = seg.analogsignals[0].transpose()
    # # number of channels and number of points
    # n_chan, n_pnts1, = len(x), len(x[0])
    # # ch_names = list()
    # # create empty data with right sizes
    # data1 = np.zeros((n_chan, 5000), dtype=float)
    # for i in range(n_chan):
    #     # if our proposed position has 1 it means the corrisponding feature is enabled,
    #     #  otherwise we don't want that channel
    #     if (position[i] == 1):
    #         data1[i, 0:5000] = x[i][0:5000]
    #     else:
    #         data1[i, 0:5000] = None

    # sfreq = int(seg.analogsignals[0].sampling_rate.magnitude)
    # # kinda useless, i just write the number of electrode to comply with previous code and fill the create_info function
    # ch_names = ['0', '1', '2', '3', '4', '5', '6', '7',
    #             '8', '9', '10', '11', '12', '13', '14', '15']
    # info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
    # # create array with mne library
    # raw = mne.io.RawArray(data1, info)


# ugly code sorry
for i, elem in enumerate(data_list):
    if i < len(data_list) - 1:
        data_list[0] = np.append(data_list[0], data_list[i + 1])

data_merge = np.zeros((n_chan, 5000 * len(files)), dtype=float)
for i in range(n_chan):
    data_merge[i] = data_list[0][i]


indices = np.arange(len(labels))
np.random.shuffle(indices)
labels = labels[indices]
for i in range(n_chan):
    x = data_merge[i]
    print(x[indices])
    data_merge[i] = x[indices]

print(len(data_merge[0]))


# for j in range(n_chan):
#     x = data_merge[j]
#     data_merge[j] = x[indices]


# data_merge, labels = shuffle(data_merge, labels)

train = data_merge.transpose()[0:25000]
test = data_merge.transpose()[25000 + 1 :]

labels_train = labels[0:25000]
labels_test = labels[25000 + 1 :]


clf = make_pipeline(StandardScaler(), SVC(kernel="linear", gamma="auto"))
clf.fit(train, labels_train)

y_pred = clf.predict(train)


print("Accuracy:", metrics.accuracy_score(labels_train, y_pred))

# for i in corr_list:
#     f.write("Score  " + str(i) + "\n")
# f.write("Now sorted...  " + "\n")
# corr_list.sort()
# for i in corr_list:
#     f.write("Score  " + str(i) + "\n")
# f.close()

# print("Applying CSP")
# csp = CSP(n_components=10, reg=None, log=None, norm_trace=False)
# x_csp = csp.fit_transform(data_merge, labels)
# x_csp = np.expand_dims(x_csp, axis=2)
# ica = UnsupervisedSpatialFilter(
#     FastICA(10, whiten="unit-variance"), average=False)
# x_csp = ica.fit_transform(data_merge)
# # pca = UnsupervisedSpatialFilter(PCA(10), average=False)
# # x_csp = pca.fit_transform(data_merge)
# x_csp = np.squeeze(x_csp, axis=2)
# print(x_csp.shape)

# Shuffle data and labels
# indices = np.arange(len(labels_list))
# np.random.shuffle(indices)
# x_csp = x_csp[indices, :, :]
# labels = np.array(labels_list)

# # Train SVM
# clf = sklearn.svm.LinearSVC(dual=False)
# print("Start training...")
# clf.fit(train, labels_train)

# print("Done Training")
# # Predict on training set
# y_pred_train = clf.predict(train)
# print("Training Accuracy:", metrics.accuracy_score(
#     labels_train, y_pred_train))

# # Predict on testing set
# y_pred_test = clf.predict(test)
# var = metrics.accuracy_score(labels_test, y_pred_test)
# print("Testing Accuracy:", var)

# # hdist = var * 0.6 + 0.2 * ((20-16)/19) + 0.2 * slack_var
# # return hdist

# print(x_csp.shape)
# print(labels.shape)

# # Split into training and testing sets
# train_size = 0.90
# train, test, labels_train, labels_test = train_test_split(
#     x_csp, labels, train_size=train_size)

# # # Standardize data
# # scaler = StandardScaler()
# # train_scaled = scaler.fit_transform(train)
# # test_scaled = scaler.transform(test)

# # Train SVM
# clf = sklearn.svm.LinearSVC(dual=False)
# print("Start training...")
# clf.fit(train, labels_train)

# print("Done Training")
# # Predict on training set
# y_pred_train = clf.predict(train)
# print("Training Accuracy:", metrics.accuracy_score(
#     labels_train, y_pred_train))

# # Predict on testing set
# y_pred_test = clf.predict(test)
# var = metrics.accuracy_score(labels_test, y_pred_test)
# print("Testing Accuracy:", var)

# hdist = var * 0.6 + 0.2 * ((20-16)/19) + 0.2 * slack_var
# return hdist


###################################################################################################    27/2/24

# num_sensors = 0
# their code #################
# It's dealing with files that Elisa's old code generated, so we have to adequate it
# sample=sample[0,:,0:16]
# their code #################
# seg = bl.segments
# print(seg)
# # two signals, we use first with 16 channels
# x = seg.analogsignals
# print(x)
# raw = mne.io.read_raw_nsx(
#     patho + str(animal_num) + fil + "/" + file, preload=True)
# raw.describe()
# sfreq = 100
# # apply a highpass filter from 1 Hz upwards
# # replace baselining with high-pass
# # test divide by 4
# # raw = raw/4
# # raw._data *= 1e+3
# # raw.filter(l_freq=800, h_freq=None)
# # raw.filter(l_freq=800, h_freq=2500)

# # raw.resample(5000)
# raw.drop_channels(["flexsensor", "pressuresensor", "motors"])
# # raw.compute_psd(fmax=50).plot(
# #     picks="data", exclude="bads", amplitude=False)

# # ica = mne.preprocessing.ICA(
# #     n_components=4, random_state=97, max_iter=800)
# # ica.fit(raw)
# # raw.load_data()
# # ica.apply(raw)

# time_on = 0.5  # definiti dal sensore codice davide da vedere, cambiano da animale ad animale e da classe a classe
# time_off = 4  # definiti del sensore
# # finestra lunga 2.5 secondi circa nel dataset ma ogni tanto
# # da usare finestre di 100ms
# x = []
# x = raw.get_data()
# x = x[0]
# for j in range(intervals.shape[0]):
#     for h in range(intervals.shape[1]):
#         for k in range(intervals.shape[2]):
#             print(intervals[j, h, :])
#             one = int(intervals[j, h, 2*k])
#             two = int(intervals[j, h, (2*k)+1])
#             print(one)
#             print(two)
#             result = x[one:two]
#             fig, axd = plt.subplot_mosaic(
#                 [["image", "density"],
#                  ["EEG", "EEG"]],
#                 layout="constrained",
#                 # "image" will contain a square image. We fine-tune the width so that
#                 # there is no excess horizontal or vertical margin around the image.
#                 width_ratios=[1.05, 2],
#             )
#             plt.plot(result)
#             plt.show()

# value_to_filter = 0.000030
# x = x[:, :int(len(x[0])*0.01)]
# rows_to_del = []
# for j in range(len(x)):
#     if position[j] != 1 and j < 16:
#         rows_to_del.append(j)
#     # if (j > 15):
#     #     rows_to_del.append(j)
#     elif position[j] == 1 and j < 16:
#         num_electrodes += 1
#     # elif (j > 15):
#     # num_sensors += 1
# num_features = num_electrodes
# print(len(x[1]))
# x = np.delete(x, rows_to_del, 0)

# fig, axd = plt.subplot_mosaic(
#     [["image", "density"],
#      ["EEG", "EEG"]],
#     layout="constrained",
#     # "image" will contain a square image. We fine-tune the width so that
#     # there is no excess horizontal or vertical margin around the image.
#     width_ratios=[1.05, 2],
# )
# for i in range(16):
#     plt.plot(x[i])
#     plt.show()

# for t in range(len(x)):
#     for y in range(len(x[t])):
#         # print(x[t, y])
#         # x[t, y] = int(x[t, y] / 4)
#         if (abs(x[t, y]) > 0.0000030):
#             counter += 1
#             x[t, y] = 0.0000030

# print(counter)
# for t in range(len(x)):
#     for y in range(len(x[t])):
# print(x[t, y])

# values = x.reshape(-1, x.shape[-1])
# scaler = StandardScaler()
# scaler = scaler.fit(values)
# x = scaler.transform(values).reshape(x.shape)

# for t in range(len(x)):
#     for y in range(len(x[t])):
#         if (abs(x[t, y]) > 0.0000030):
#             x[t, y] = 0

# print("cleaned")

# create a numpy array of EEG data from the MNE raw object
# eeg_array = x[:, 0:len(x[0])]

# cic = int(len(x[0]) / sfreq)

# extract the sampling frequency from the MNE raw object

# (optional) make sure your asr is only fitted to clean parts of the data
# pre_cleaned, _ = clean_windows(x, sfreq, max_bad_chans=0.1)
# fit the asr
# M, T = asr_calibrate(x, sfreq, cutoff=15)
# # apply it
# clean_array = asr_process(x, sfreq, M, T)

# widths = np.arange(1, (sfreq * 2) + 1)
# datas = np.zeros((cic, (sfreq * 2), sfreq, num_features), dtype=float)
# for i in range(num_features):
#     for j in range(cic):
#         datas[j, :, :, i], freq = pywt.cwt(
#             x[i, (sfreq * j) : sfreq * (j + 1)],
#             widths,
#             "mexh",
#             sampling_period=1 / 5000,
#         )
# print(datas.shape)
# data_list.append(datas)
# labels_list.extend([counts] * (cic))

# here we call correlation
# correlate_function(num_features, 10, other_list, f)


###LEARN SOMETRHIGN

# sorted_indices = sorted(range(len(mav)), key=lambda i: mav[i], reverse=True)[:x]


import neo
import os
from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
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
from correlation import correlate_function
import pywt


tfk = tf.keras
tfkl = tf.keras.layers


def print_value_counts(arr):
    count_dict = {}
    for value in arr:
        if value in count_dict:
            count_dict[value] += 1
        else:
            count_dict[value] = 1

    return count_dict


# def eval(position):
# s
animal_num = input("Enter animal number 1,2 or 3 \n")
animal_num = animal_num + "/"
patho = "../src/data/animal "
path_arrays = "../src/numpy_arrays/animal "
patho_nev = "../src/data_nev/animal "
animal_number_folders = [f for f in os.listdir(path_arrays + str(animal_num))]
anima_number_folders_nev = [f for f in os.listdir(patho_nev + str(animal_num))]


# mat = scipy.io.loadmat("../100ms 2/sample_0002_prop_angle_-10_inpoints_0.444_002.mat")
files_mat = [f for f in os.listdir("../100ms/")]


data_list = []
labels_list = []
labels_list_windowless = []
counter = 0
all_classes = []
all_classes_windowless = []
correlation_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
channel_bool = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]


f = open("results.txt", "a")
current_time = datetime.now()
for i in range(0, 16):
    channel_bool[i] = 1
    print("io")
for i in files_mat:
    mat = scipy.io.loadmat("../100ms/" + i)
    data_mat = mat["to_save"]
    all_classes.append(data_mat.transpose())
    if "_prop_angle_-30" in i or "_prop_angle_-20" in i or "_prop_angle_-10" in i:
        labels_list.extend([0] * 1)
    elif "noci_trial" in i:
        labels_list.extend([2] * 1)
    elif "_touch_" in i:
        labels_list.extend([3] * 1)
    else:
        labels_list.extend([1] * 1)
# for iteration_classes, classes in enumerate(animal_number_folders):
#     # f.write(str(counts) + " is " + fil + "\n")
#     print(classes)
#     if classes == ".DS_Store":
#         continue
#     files = [f for f in os.listdir(path_arrays + str(animal_num) + classes)]
#     one_class_list = []
#     one_class_list_window = []
#     for file_count, file in enumerate(files):
#         # num_electrodes = 0
#         print(file)
#         if file == ".DS_Store":
#             continue
#         temp = np.load(
#             "numpy_arrays/" + "animal " + animal_num + classes + "/" + file,
#             allow_pickle=True,
#         )

#         length_of_temp = len(temp[0])
#         window = 500
#         division = int(length_of_temp / window)
#         # extract the sampling frequency from the MNE raw object
#         # (optional) make sure your asr is only fitted to clean parts of the data
#         # widths = np.arange(1, (window * 1) + 1)
#         # datas = np.zeros((div, window, window, 16), dtype=float)
#         widtho = 17
#         widths = np.arange(1, (widtho * 1) + 1)
#         datas = np.zeros((division, window, 16), dtype=float)
#         # for i in range(16):
#         if not (
#             (classes == "touch" and file_count % 24 != 0)
#             or (classes == "prop" and file_count % 4 != 0)
#             or (classes == "prop +" and file_count % 4 != 0)
#             or (classes == "noci" and file_count % 3 != 0)
#             # or (classes == "noci" and file_count % 5 != 0)
#         ):
#         for j in range(division):
#             datas[j, :, :] = temp[:, (window * j) : window * (j + 1)].transpose()
#         # if not (fil == "touch" and count % 14 != 0):
#         # for j in range(division):
#         #     for i in range(16):
#         #         datas[j, :, :, i] = scipy.signal.cwt(
#         #             temp[i, (window * j) : window * (j + 1)],
#         #             scipy.signal.ricker,
#         #             widths,
#         #         )

#         # print(datas.shape)
#         # data_list.append(datas)
#         # print(len(temp[0]))

#         # we keep track of both the version with window and without window
#         labels_list_windowless.extend([iteration_classes] * len(temp[0]))
#         labels_list.extend([iteration_classes] * (division))
#         # list to keep data of one classes, then we concatenate
#         one_class_list.append(datas)
#         one_class_list_window.append(temp.transpose())

# # concatenation to combine the lists and then conversion to numpy array
# inter_step = np.concatenate(one_class_list, axis=0)
# inter_window = np.concatenate(one_class_list_window, axis=0)
# all_classes.append(inter_step)
# all_classes_windowless.append(inter_window)

print(len(all_classes))
labels = np.array(labels_list)
# data_merge = np.concatenate(all_classes, axis=0)
# data_merge_windowless = np.concatenate(all_classes_windowless, axis=0)
# data_merge_windowless = data_merge_windowless.transpose()

data_merge = np.array(all_classes)
# data_merge = data_merge.transpose()
print(data_merge.shape)
# print(data_merge_windowless.shape)
print(labels)
labels = tfk.utils.to_categorical(labels)
print(labels.shape)

num_electrodes = 0
rows_to_del = []
for j in range(len(channel_bool)):
    if channel_bool[j] != 1 and j < 16:
        rows_to_del.append(j)
    elif channel_bool[j] == 1 and j < 16:
        num_electrodes += 1

num_features = num_electrodes
data_merge = np.delete(data_merge, rows_to_del, 0)
# data_merge_windowless = np.delete(data_merge_windowless, rows_to_del, 0)
print("length of data_merge... " + str(len(data_merge[1])))

labels_correlation_windowless = print_value_counts(labels_list_windowless)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(labels_list), y=labels_list
)
print(labels_correlation_windowless)

# Convert class weights to a dictionary
class_weights_dict = dict(enumerate(class_weights))

# class_weights_dict[0] = class_weights_dict[0] * 1.5
# class_weights_dict[3] = class_weights_dict[3] * 1.2
print(class_weights_dict)

train_ratio = 0.80
validation_ratio = 0.20
test_ratio = 0.20

k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True)

x_train, x_test, y_train, y_test = train_test_split(
    data_merge, labels, train_size=train_ratio, shuffle=True
)

# values = x_train.reshape(-1, x_train.shape[-1])
# scaler = StandardScaler()
# scaler = scaler.fit(values)
# x_train_norm = scaler.transform(values).reshape(x_train.shape)
# # Standardization Validation
# # values_val = x_train.reshape(-1, x_train.shape[-1])
# # x_val_norm = scaler.transform(values_val).reshape(x_train.shape)
# # Standardization Test
# values_test = x_test.reshape(-1, x_test.shape[-1])
# x_test_norm = scaler.transform(values_test).reshape(x_test.shape)

# correlate_function(
#     num_features,
#     4,
#     data_merge_windowless,
#     labels_correlation_windowless,
#     f,
#     window,
#     correlation_scores,
# )

print("YO")
print(len(labels_list_windowless))
print(correlation_scores)
print("YO")

# x_train_less, x_test_less, y_train_less, y_test_less = train_test_split(
#     data_merge_windowless.transpose(),
#     labels_list_window,
#     train_size=train_ratio,
#     shuffle=True,
#     stratify=labels_list_window,
# )
into = 0
for train_index, val_index in kf.split(x_train):
    into += 1
    model = ENGNet2(
        4,
        num_features,
    )
    model_name = "ENGNet2"
    model.summary()
    # print("ciao")
    # print(train_index)
    # print(val_index)
    # print("ciao")
    x_train_k, x_val = x_train[train_index], x_train[val_index]
    y_train_k, y_val = y_train[train_index], y_train[val_index]

    # x_train, x_val, labels_train, labels_val = train_test_split(
    #     x_train,
    #     labels_train,
    #     train_size=1 - validation_ratio,
    #     random_state=420,
    #     stratify=labels_train,
    # )

    # print(num_features)
    # print(sfreq)
    # model = ENGNet2(
    #     4, num_features, x_train_norm.shape[0], window, class_weights_dict
    # )
    # model_name = "ENGNet2"

    # model.summary()

    layer_to_save_weights = model.layers[2]
    weights_to_save = layer_to_save_weights.get_weights()

    output_folder_cv = "../"

    checkpoint_path = os.path.join(output_folder_cv, "best_model_checkpoint.h5")
    model_checkpoint = tfk.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_f1_score",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    # Save the weights to a file
    weights_file_path = str(into) + "saved_weights.h5"
    with tf.keras.utils.CustomObjectScope(
        {"GlorotUniform": tf.keras.initializers.GlorotUniform}
    ):
        layer_to_save_weights.set_weights(weights_to_save)
        model.save_weights(weights_file_path)
    print(x_train.shape)
    print(y_train_k.shape)
    history = model.fit(
        x=x_train_k,
        y=y_train_k,
        epochs=45,
        validation_data=(x_val, y_val),
        callbacks=[
            tfk.callbacks.EarlyStopping(
                monitor="val_f1_score",
                mode="max",
                patience=30,
                restore_best_weights=True,
            ),
            tfk.callbacks.ReduceLROnPlateau(
                monitor="val_f1_score", mode="max", patience=30, factor=0.5
            ),
            model_checkpoint,
        ],
    ).history

    model.load_weights(checkpoint_path)

    # Plotting, da aggiiungere la loss
    best_epoch = np.argmax(history["val_f1_score"])
    plt.figure(figsize=(17, 4))
    plt.plot(history["loss"], label="Training loss", alpha=0.8, color="#ff7f0e")
    plt.plot(history["val_loss"], label="Validation loss", alpha=0.9, color="#5a9aa5")
    # plt.axvline(x=best_epoch, label="Best epoch", alpha=0.3, ls="--", color="#5a9aa5")
    plt.title("Categorical Crossentropy")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.show()

    plt.figure(figsize=(17, 4))
    plt.plot(
        history["val_f1_score"],
        label="Training accuracy",
        alpha=0.8,
        color="#ff7f0e",
    )
    plt.plot(
        history["val_f1_score"],
        label="Validation accuracy",
        alpha=0.9,
        color="#5a9aa5",
    )
    # plt.axvline(x=best_epoch, label="Best epoch", alpha=0.3, ls="--", color="#5a9aa5")
    plt.title("F1Score")
    plt.legend()
    plt.grid(alpha=0.3)

    img_name = (
        str(current_time)
        + "_"
        + str(num_features)
        + "_"
        + model_name
        + "_"
        + str(model.count_params())
        + ".png"
    )
    file_path = os.path.join("img_results", img_name)

    plt.savefig(file_path)

    # plt.show()

    plt.figure(figsize=(18, 3))
    plt.plot(history["lr"], label="Learning Rate", alpha=0.8, color="#ff7f0e")
    # plt.axvline(x=best_epoch, label="Best epoch", alpha=0.3, ls="--", color="#5a9aa5")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.show()

    model.save("on_a_gang_model" + str(into))
    prediction = model.predict(x_test)
    cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(prediction, axis=-1))

    # Compute the classification metrics
    accuracy = accuracy_score(
        np.argmax(y_test, axis=-1), np.argmax(prediction, axis=-1)
    )
    precision = precision_score(
        np.argmax(y_test, axis=-1),
        np.argmax(prediction, axis=-1),
        average="macro",
    )
    recall = recall_score(
        np.argmax(y_test, axis=-1),
        np.argmax(prediction, axis=-1),
        average="macro",
    )
    f1 = f1_score(
        np.argmax(y_test, axis=-1),
        np.argmax(prediction, axis=-1),
        average="macro",
    )
    print("Accuracy:", accuracy.round(4))
    print("Precision:", precision.round(4))
    print("Recall:", recall.round(4))
    print("F1:", f1.round(4))

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    x_axis_labels = ["prop-", "prop+", "touch", "noci "]
    y_axis_labels = ["prop-", "prop+", "touch", "noci "]
    sns.heatmap(
        cm.T,
        annot=True,
        cmap="Blues",
        fmt="d",
        annot_kws={"size": 12},
        xticklabels=x_axis_labels,
        yticklabels=y_axis_labels,
    )
    plt.xlabel("True labels")
    plt.ylabel("Predicted labels")

    img_name = (
        str(current_time)
        + "_"
        + str(num_features)
        + "_"
        + model_name
        + "_"
        + str(model.count_params())
        + "_conf"
        + ".png"
    )
    file_path = os.path.join("img_results", img_name)

    plt.savefig(file_path)

model1 = tf.keras.models.load_model("on_a_gang_model" + "1")
model2 = tf.keras.models.load_model("on_a_gang_model" + "2")
model3 = tf.keras.models.load_model("on_a_gang_model" + "3")
model4 = tf.keras.models.load_model("on_a_gang_model" + "4")
model5 = tf.keras.models.load_model("on_a_gang_model" + "5")

predictions1 = model1.predict(x_test)
predictions2 = model2.predict(x_test)
predictions3 = model3.predict(x_test)
predictions4 = model4.predict(x_test)
predictions5 = model5.predict(x_test)
# predictions.shape

# Assuming you have predictions from 5 models stored in a list
model_predictions = [
    predictions1,
    predictions2,
    predictions3,
    predictions4,
    predictions5,
]

# Convert predictions to numpy arrays
model_predictions_np = [np.array(pred) for pred in model_predictions]
print(model_predictions_np)
# Combine predictions by averaging
combined_predictions = np.mean(model_predictions_np, axis=0)
print(combined_predictions)

# Take the class with the highest average prediction
final_prediction = combined_predictions
print(final_prediction)

cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(final_prediction, axis=-1))

# Compute the classification metrics
accuracy = accuracy_score(
    np.argmax(y_test, axis=-1), np.argmax(final_prediction, axis=-1)
)
precision = precision_score(
    np.argmax(y_test, axis=-1),
    np.argmax(final_prediction, axis=-1),
    average="macro",
)
recall = recall_score(
    np.argmax(y_test, axis=-1),
    np.argmax(final_prediction, axis=-1),
    average="macro",
)
f1 = f1_score(
    np.argmax(y_test, axis=-1),
    np.argmax(final_prediction, axis=-1),
    average="macro",
)
print("Accuracy:", accuracy.round(4))
print("Precision:", precision.round(4))
print("Recall:", recall.round(4))
print("F1:", f1.round(4))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm.T, annot=True, cmap="Blues", fmt="d", annot_kws={"size": 12})
plt.xlabel("True labels")
plt.ylabel("Predicted labels")

img_name = (
    str(current_time)
    + "_"
    + str(num_features)
    + "_"
    + model_name
    + "_"
    + str(model.count_params())
    + "_conf"
    + ".png"
)
file_path = os.path.join("img_results", img_name)

plt.savefig(file_path)
# plt.show()

f.write(
    str(current_time)
    + "_"
    + str(num_features)
    + "_"
    + model_name
    + "_"
    + str(model.count_params())
    + " file"
    + "\n"
)
for i in channel_bool:
    f.write(str(i))

f.write("\n")
f.write(str(accuracy))
f.write("\n")
f.write(str(precision))
f.write("\n")
f.write(str(recall))
f.write("\n")
f.write(str(f1))

f.write("\n")
f.write("\n")
f.write("\n")
# f.close()






################################




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
from correlation import correlate_function
import pywt


tfk = tf.keras
tfkl = tf.keras.layers


def print_value_counts(arr):
    count_dict = {}
    for value in arr:
        if value in count_dict:
            count_dict[value] += 1
        else:
            count_dict[value] = 1

    return count_dict


def get_label_from_path(name):
    # Remove file extension
    label = name.replace(".mat", "")
    # Remove intial part of file name, eg "sample_0161_"
    label = label[12:]
    return label


# Encodes label depending on the name of the file
def encod_label(lab):
    if lab[0:4] == "noci" or lab[0:4] == "Noci" or lab[0:5] == "Pinch":
        label = 3
    elif lab[0:4] == "prop" and lab[11] == "-":
        label = 0
    elif lab[0:4] == "prop":
        label = 1
    elif lab[0:5] == "touch":
        label = 2
    return label


def import_sample(path):
    mat = scipy.io.loadmat(path, mat_dtype=True)
    x = np.array(mat["to_save"])
    # file_idx=int(mat['file_idx'])
    # sample_idx=int(mat['sample_idx'])
    # original_sample=int(mat['original_sample'])
    # n_samples_original=int(mat['n_samples_original'])
    return x  # ,file_idx,sample_idx,original_sample,n_samples_original


def transform_data_not_transpose(file_name, file_paths, n_features):
    y_samp = np.empty(len(file_name))
    # file_idx_vec=np.empty(len(file_name))
    # sample_idx_vec=np.empty(len(file_name))
    # original_sample_vec=np.empty(len(file_name))
    # n_samples_original_vec=np.empty(len(file_name))
    sample = []
    for file_number in range(len(file_name)):

        # file_idx,sample_idx,original_sample,n_sample_original

        # print(file_number)
        sample = import_sample(file_paths[file_number])

        if len(np.shape(sample)) == 3:
            # It's dealing with files that Elisa's old code generated, so we have to adequate it
            sample = sample[0, :, 0:16]
        else:
            # Dealing with files that Elisa's new code, changed by Rafael, generated
            sample = np.transpose(sample)
        if file_number == 0:
            x_samp = np.empty((len(file_name), np.shape(sample)[0], n_features))

        # print(file_name[file_number])
        # Gets the label from the file name
        lab = get_label_from_path(file_name[file_number])
        # print(lab)
        # Stores the sample
        x_samp[file_number, :, :] = sample

        # Encodes the label to a value that will be the target
        y_samp[file_number] = encod_label(lab)

        # Encodes if the sample was obtained via overlapping or not and the characteristics of it so
        # The training set can be built with no overlapping regarding the test set
        # file_idx_vec[file_number]=file_idx
        # sample_idx_vec[file_number]=sample_idx
        # original_sample_vec[file_number]=original_sample
        # n_samples_original_vec[file_number]=n_samples_original

        # x_samp_trasposta = np.transpose(x_samp, (0, 2, 1))

        # , file_idx_vec, sample_idx_vec, original_sample_vec, n_samples_original_vec

    return x_samp, y_samp


# def eval(position):
# s
# animal_num = input("Enter animal number 1,2 or 3 \n")
# animal_num = animal_num + "/"
# patho = "../src/data/animal "
# path_arrays = "../src/numpy_arrays/animal "
# patho_nev = "../src/data_nev/animal "
# animal_number_folders = [f for f in os.listdir(path_arrays + str(animal_num))]
# anima_number_folders_nev = [f for f in os.listdir(patho_nev + str(animal_num))]


# mat = scipy.io.loadmat("../100ms 2/sample_0002_prop_angle_-10_inpoints_0.444_002.mat")
files_mat = [f for f in os.listdir("../100ms/")]

test_results = {
    "test_accuracy": [],
    "test_f1_score": [],
    "test_weighted_f1_score": [],
    "test_confusion_matrix": [],
}

data_list = []
labels_list = []
labels_list_windowless = []
counter = 0
all_classes = []
all_classes_windowless = []
correlation_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
channel_bool = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
num_features = 16
check = 0
f = open("results.txt", "a")
current_time = datetime.now()
# for i in range(0, 16):
#     channel_bool[i] = 1
#     # print("io")
# for i, m in enumerate(files_mat):
#     mat = scipy.io.loadmat("../100ms/" + m)
#     data_mat = mat["to_save"]
#     labo = get_label_from_path(m)
#     laboo = encod_label(labo)
#     all_classes.append(data_mat.transpose())
#     labels_list.extend([laboo] * 1)
#     labels_list_windowless.extend([laboo] * len(data_mat[0]))

path_folder = "../100ms/"
file_name = [f for f in os.listdir("../100ms/")]
file_paths = []
for file_number in range(len(file_name)):
    file = os.path.join(path_folder, file_name[file_number])
    file_paths.append(file)


x_samp, y_samp = transform_data_not_transpose(file_name, file_paths, num_features)


print(x_samp.shape)

# print(len(all_classes))
# labels = np.array(labels_list)
# # data_merge = np.concatenate(all_classes, axis=0)
# # data_merge_windowless = np.concatenate(all_classes_windowless, axis=0)
# # data_merge_windowless = data_merge_windowless.transpose()

# data_merge = np.array(all_classes)
# print(data_merge.shape)

# # data_merge = data_merge.transpose()
# print(data_merge.shape)
# data_merge_windowless = np.concatenate(data_merge, axis=0)
# data_merge_windowless = data_merge_windowless.transpose()
# print(data_merge_windowless.shape)
# # print(data_merge_windowless.shape)
# print(labels)
# labels = tfk.utils.to_categorical(labels)
# print(labels.shape)

# num_electrodes = 0
# rows_to_del = []
# for j in range(len(channel_bool)):
#     if channel_bool[j] != 1 and j < 16:
#         rows_to_del.append(j)
#     elif channel_bool[j] == 1 and j < 16:
#         num_electrodes += 1

# num_features = num_electrodes
# data_merge = np.delete(data_merge, rows_to_del, 0)
# # data_merge_windowless = np.delete(data_merge_windowless, rows_to_del, 0)
# print("length of data_merge... " + str(len(data_merge[1])))

# labels_correlation_windowless = print_value_counts(labels_list_windowless)

# Compute class weights
# class_weights = compute_class_weight(
#     class_weight="balanced", classes=np.unique(labels_list), y=labels_list
# )
# print(labels_correlation_windowless)

# # Convert class weights to a dictionary
# class_weights_dict = dict(enumerate(class_weights))

# class_weights_dict[0] = class_weights_dict[0] * 1.5
# class_weights_dict[3] = class_weights_dict[3] * 1.2
# print(class_weights_dict)

train_ratio = 0.80
validation_ratio = 0.20
test_ratio = 0.20

k = 5  # Number of folds
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

x_samp_2 = np.empty((len(file_name), 16, 500))

for i, j in enumerate(x_samp):
    x_samp_2[i] = j.transpose()

x_train, x_test, y_train, y_test = train_test_split(
    x_samp_2, y_samp, train_size=train_ratio, shuffle=True, random_state=42
)

# values = x_train.reshape(-1, x_train.shape[-1])
# scaler = StandardScaler()
# scaler = scaler.fit(values)
# x_train_norm = scaler.transform(values).reshape(x_train.shape)
# # Standardization Validation
# # values_val = x_train.reshape(-1, x_train.shape[-1])
# # x_val_norm = scaler.transform(values_val).reshape(x_train.shape)
# # Standardization Test
# values_test = x_test.reshape(-1, x_test.shape[-1])
# x_test_norm = scaler.transform(values_test).reshape(x_test.shape)

# correlate_function(
#     num_features,
#     4,
#     data_merge_windowless,
#     labels_correlation_windowless,
#     f,
#     500,
#     correlation_scores,
# )

# print("YO")
# print(len(labels_list_windowless))
# print(correlation_scores)
# print("YO")

# x_train_less, x_test_less, y_train_less, y_test_less = train_test_split(
#     data_merge_windowless.transpose(),
#     labels_list_window,
#     train_size=train_ratio,
#     shuffle=True,
#     stratify=labels_list_window,
# )
into = 0
for train_index, val_index in kf.split(x_train, y_train):
    into += 1
    model = ENGNet2(
        4,
        num_features,
    )
    model_name = "ENGNet2"
    model.summary()
    # print("ciao")
    # print(train_index)
    # print(val_index)
    # print("ciao")
    x_train_k, x_val = x_train[train_index], x_train[val_index]
    y_train_k, y_val = y_train[train_index], y_train[val_index]

    # x_train, x_val, labels_train, labels_val = train_test_split(
    #     x_train,
    #     labels_train,
    #     train_size=1 - validation_ratio,
    #     random_state=420,
    #     stratify=labels_train,
    # )

    # print(num_features)
    # print(sfreq)
    # model = ENGNet2(
    #     4, num_features, x_train_norm.shape[0], window, class_weights_dict
    # )
    # model_name = "ENGNet2"

    # model.summary()

    layer_to_save_weights = model.layers[2]
    weights_to_save = layer_to_save_weights.get_weights()

    output_folder_cv = "../"

    # checkpoint_path = os.path.join(output_folder_cv, "best_model_checkpoint.h5")
    # model_checkpoint = tfk.callbacks.ModelCheckpoint(
    #     checkpoint_path,
    #     monitor="val_f1_score",
    #     save_best_only=True,
    #     save_weights_only=True,
    #     verbose=1,
    # )

    # Save the weights to a file
    # weights_file_path = str(into) + "saved_weights.h5"
    # with tf.keras.utils.CustomObjectScope(
    #     {"GlorotUniform": tf.keras.initializers.GlorotUniform}
    # ):
    #     layer_to_save_weights.set_weights(weights_to_save)
    #     model.save_weights(weights_file_path)
    print(x_train.shape)
    print(y_train_k.shape)

    history = model.fit(
        x=x_train_k,
        y=y_train_k,
        epochs=50,
        validation_data=(x_val, y_val),
        # class_weight=class_weights_dict,
        callbacks=[
            tfk.callbacks.EarlyStopping(
                monitor="accuracy",
                mode="max",
                patience=15,
                restore_best_weights=True,
            ),
            tfk.callbacks.ReduceLROnPlateau(
                monitor="accuracy", mode="max", patience=15, factor=0.5
            ),
            # model_checkpoint,
        ],
    ).history

    # model.load_weights(checkpoint_path)

    # Plotting, da aggiiungere la loss
    best_epoch = np.argmax(history["accuracy"])
    plt.figure(figsize=(17, 4))
    plt.plot(history["loss"], label="Training loss", alpha=0.8, color="#ff7f0e")
    plt.plot(history["val_loss"], label="Validation loss", alpha=0.9, color="#5a9aa5")
    # plt.axvline(x=best_epoch, label="Best epoch", alpha=0.3, ls="--", color="#5a9aa5")
    plt.title("Categorical Crossentropy")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.show()

    plt.figure(figsize=(17, 4))
    plt.plot(
        history["accuracy"],
        label="Training accuracy",
        alpha=0.8,
        color="#ff7f0e",
    )
    plt.plot(
        history["accuracy"],
        label="Validation accuracy",
        alpha=0.9,
        color="#5a9aa5",
    )
    # plt.axvline(x=best_epoch, label="Best epoch", alpha=0.3, ls="--", color="#5a9aa5")
    plt.title("F1Score")
    plt.legend()
    plt.grid(alpha=0.3)

    img_name = (
        str(current_time)
        + "_"
        + str(num_features)
        + "_"
        + model_name
        + "_"
        + str(model.count_params())
        + ".png"
    )
    file_path = os.path.join("img_results", img_name)

    plt.savefig(file_path)

    # plt.show()

    plt.figure(figsize=(18, 3))
    plt.plot(history["lr"], label="Learning Rate", alpha=0.8, color="#ff7f0e")
    # plt.axvline(x=best_epoch, label="Best epoch", alpha=0.3, ls="--", color="#5a9aa5")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.show()

    model.save("on_a_gang_model" + str(into))
    y_val_pred = model.predict(x_val)
    y_val_pred_nohot = np.argmax(y_val_pred, axis=1)
    conf_matrix_val = confusion_matrix(y_val, y_val_pred_nohot)
    # cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(prediction, axis=-1))

    # Compute the classification metrics
    # Calculate testing metrics
    y_test_pred = model.predict(x_test)
    y_test_pred_nohot = np.argmax(y_test_pred, axis=1)

    # Calculate testing metrics
    test_results["test_accuracy"].append(accuracy_score(y_test, y_test_pred_nohot))
    test_results["test_f1_score"].append(
        f1_score(y_test, y_test_pred_nohot, average="macro")
    )
    test_results["test_weighted_f1_score"].append(
        f1_score(y_test, y_test_pred_nohot, average="weighted")
    )
    test_results["test_confusion_matrix"].append(
        confusion_matrix(y_test, y_test_pred_nohot)
    )

    print("Test Accuracy:", test_results["test_accuracy"])
    print("Test F1 score:", test_results["test_f1_score"])
    print("Test weighted F1 score:", test_results["test_weighted_f1_score"])

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    x_axis_labels = ["prop-", "prop+", "touch", "noci "]
    y_axis_labels = ["prop-", "prop+", "touch", "noci "]
    sns.heatmap(
        conf_matrix_val.T,
        annot=True,
        cmap="Blues",
        fmt="d",
        annot_kws={"size": 12},
        xticklabels=x_axis_labels,
        yticklabels=y_axis_labels,
    )
    plt.xlabel("True labels")
    plt.ylabel("Predicted labels")

    img_name = (
        str(current_time)
        + "_"
        + str(num_features)
        + "_"
        + model_name
        + "_"
        + str(model.count_params())
        + "_conf"
        + ".png"
    )
    file_path = os.path.join("img_results", img_name)

    plt.savefig(file_path)

model1 = tf.keras.models.load_model("on_a_gang_model" + "1")
model2 = tf.keras.models.load_model("on_a_gang_model" + "2")
model3 = tf.keras.models.load_model("on_a_gang_model" + "3")
model4 = tf.keras.models.load_model("on_a_gang_model" + "4")
model5 = tf.keras.models.load_model("on_a_gang_model" + "5")

predictions1 = model1.predict(x_test)
predictions2 = model2.predict(x_test)
predictions3 = model3.predict(x_test)
predictions4 = model4.predict(x_test)
predictions5 = model5.predict(x_test)
# predictions.shape

# Assuming you have predictions from 5 models stored in a list
model_predictions = [
    predictions1,
    predictions2,
    predictions3,
    predictions4,
    predictions5,
]

# Convert predictions to numpy arrays
model_predictions_np = [np.array(pred) for pred in model_predictions]
print(model_predictions_np)
# Combine predictions by averaging
combined_predictions = np.mean(model_predictions_np, axis=0)
print(combined_predictions)

# Take the class with the highest average prediction
final_prediction = combined_predictions
print(final_prediction)

cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(final_prediction, axis=-1))

# Compute the classification metrics
accuracy = accuracy_score(
    np.argmax(y_test, axis=-1), np.argmax(final_prediction, axis=-1)
)
precision = precision_score(
    np.argmax(y_test, axis=-1),
    np.argmax(final_prediction, axis=-1),
    average="macro",
)
recall = recall_score(
    np.argmax(y_test, axis=-1),
    np.argmax(final_prediction, axis=-1),
    average="macro",
)
f1 = f1_score(
    np.argmax(y_test, axis=-1),
    np.argmax(final_prediction, axis=-1),
    average="macro",
)
print("Accuracy:", accuracy.round(4))
print("Precision:", precision.round(4))
print("Recall:", recall.round(4))
print("F1:", f1.round(4))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm.T, annot=True, cmap="Blues", fmt="d", annot_kws={"size": 12})
plt.xlabel("True labels")
plt.ylabel("Predicted labels")

img_name = (
    str(current_time)
    + "_"
    + str(num_features)
    + "_"
    + model_name
    + "_"
    + str(model.count_params())
    + "_conf"
    + ".png"
)
file_path = os.path.join("img_results", img_name)

plt.savefig(file_path)
# plt.show()

f.write(
    str(current_time)
    + "_"
    + str(num_features)
    + "_"
    + model_name
    + "_"
    + str(model.count_params())
    + " file"
    + "\n"
)
for i in channel_bool:
    f.write(str(i))

f.write("\n")
f.write(str(accuracy))
f.write("\n")
f.write(str(precision))
f.write("\n")
f.write(str(recall))
f.write("\n")
f.write(str(f1))

f.write("\n")
f.write("\n")
f.write("\n")
# f.close()


# import neo
# import os
# from matplotlib import pyplot as plt
# import numpy as np
# import scipy
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# import copy
# import scipy.io


# # from sklearn.svm import SVC
# import mne

# # from mne.decoding import CSP
# import asrpy
# from asrpy import asr_calibrate, asr_process, clean_windows

# # from mne.decoding import UnsupervisedSpatialFilter
# # from sklearn.decomposition import PCA, FastICA
# import tensorflow as tf
# import seaborn as sns
# from modelz import *

# # from intervals_mat import *

# from datetime import datetime
# from correlation import correlate_function
# import pywt


# tfk = tf.keras
# tfkl = tf.keras.layers


# def print_value_counts(arr):
#     count_dict = {}
#     for value in arr:
#         if value in count_dict:
#             count_dict[value] += 1
#         else:
#             count_dict[value] = 1

#     return count_dict


# # def eval(position):
# # s
# animal_num = input("Enter animal number 1,2 or 3 \n")
# animal_num = animal_num + "/"
# patho = "../src/data/animal "
# path_arrays = "../src/numpy_arrays/animal "
# patho_nev = "../src/data_nev/animal "
# animal_number_folders = [f for f in os.listdir(path_arrays + str(animal_num))]
# anima_number_folders_nev = [f for f in os.listdir(patho_nev + str(animal_num))]


# # mat = scipy.io.loadmat("../100ms 2/sample_0002_prop_angle_-10_inpoints_0.444_002.mat")
# files_mat = [f for f in os.listdir("../100ms/")]


# data_list = []
# labels_list = []
# labels_list_windowless = []
# counter = 0
# all_classes = []
# all_classes_windowless = []
# correlation_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# channel_bool = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]


# f = open("results.txt", "a")
# current_time = datetime.now()
# for i in range(0, 16):
#     channel_bool[i] = 1
#     print("io")
# # for i in files_mat:
# #     mat = scipy.io.loadmat("../100ms/" + i)
# #     data_mat = mat["to_save"]
# #     all_classes.append(data_mat.transpose())
# #     if "_prop_angle_-30" in i or "_prop_angle_-20" in i or "_prop_angle_-10" in i:
# #         labels_list.extend([0] * 1)
# #     elif "noci_trial" in i:
# #         labels_list.extend([2] * 1)
# #     elif "_touch_" in i:
# #         labels_list.extend([3] * 1)
# #     else:
# #         labels_list.extend([1] * 1)
# for iteration_classes, classes in enumerate(animal_number_folders):
#     # f.write(str(counts) + " is " + fil + "\n")
#     print(classes)
#     if classes == ".DS_Store":
#         continue
#     files = [f for f in os.listdir(path_arrays + str(animal_num) + classes)]
#     one_class_list = []
#     one_class_list_window = []
#     for file_count, file in enumerate(files):
#         # num_electrodes = 0
#         print(file)
#         if file == ".DS_Store":
#             continue
#         temp = np.load(
#             "numpy_arrays/" + "animal " + animal_num + classes + "/" + file,
#             allow_pickle=True,
#         )

#         length_of_temp = len(temp[0])
#         window = 500
#         division = int(length_of_temp / window)
#         # extract the sampling frequency from the MNE raw object
#         # (optional) make sure your asr is only fitted to clean parts of the data
#         # widths = np.arange(1, (window * 1) + 1)
#         # datas = np.zeros((div, window, window, 16), dtype=float)
#         widtho = 17
#         widths = np.arange(1, (widtho * 1) + 1)
#         datas = np.zeros((division, window, 16), dtype=float)
#         # for i in range(16):
#         if not (
#             (classes == "touch" and file_count % 1000 != 0)
#             or (classes == "prop" and file_count % 1 != 0)
#             or (classes == "prop +" and file_count % 1 != 0)
#             or (classes == "noci" and file_count % 1 != 0)
#             # or (classes == "noci" and file_count % 5 != 0)
#         ):
#             for j in range(division):
#                 datas[j, :, :] = temp[:, (window * j) : window * (j + 1)].transpose()
#             # if not (fil == "touch" and count % 14 != 0):
#             # for j in range(division):
#             #     for i in range(16):
#             #         datas[j, :, :, i] = scipy.signal.cwt(
#             #             temp[i, (window * j) : window * (j + 1)],
#             #             scipy.signal.ricker,
#             #             widths,
#             #         )

#             # print(datas.shape)
#             # data_list.append(datas)
#             # print(len(temp[0]))

#             # we keep track of both the version with window and without window
#             labels_list_windowless.extend([iteration_classes] * len(temp[0]))
#             labels_list.extend([iteration_classes] * (division))
#             # list to keep data of one classes, then we concatenate
#             one_class_list.append(datas)
#             one_class_list_window.append(temp.transpose())

#     # concatenation to combine the lists and then conversion to numpy array
#     inter_step = np.concatenate(one_class_list, axis=0)
#     inter_window = np.concatenate(one_class_list_window, axis=0)
#     all_classes.append(inter_step)
#     all_classes_windowless.append(inter_window)

# print(len(all_classes))
# labels = np.array(labels_list)
# data_merge = np.concatenate(all_classes, axis=0)
# print(data_merge.shape)

# # data_merge = data_merge.transpose()
# print(data_merge.shape)
# data_merge_windowless = np.concatenate(data_merge, axis=0)
# data_merge_windowless = data_merge_windowless.transpose()
# # print(data_merge_windowless.shape)
# # print(data_merge_windowless.shape)
# # print(labels)
# labels = tfk.utils.to_categorical(labels)
# print(labels.shape)


# # concatenation to combine the lists and then conversion to numpy array
# inter_step = np.concatenate(one_class_list, axis=0)
# inter_window = np.concatenate(one_class_list_window, axis=0)
# # all_classes.append(inter_step)
# all_classes_windowless.append(inter_window)

# print(len(all_classes))
# labels = np.array(labels_list)
# # data_merge = np.concatenate(all_classes, axis=0)
# # data_merge_windowless = np.concatenate(all_classes_windowless, axis=0)
# # data_merge_windowless = data_merge_windowless.transpose()

# # data_merge = data_merge.transpose()
# print(data_merge.shape)
# # print(data_merge_windowless.shape)
# print(labels)
# labels = tfk.utils.to_categorical(labels)
# print(labels.shape)

# num_electrodes = 0
# rows_to_del = []
# for j in range(len(channel_bool)):
#     if channel_bool[j] != 1 and j < 16:
#         rows_to_del.append(j)
#     elif channel_bool[j] == 1 and j < 16:
#         num_electrodes += 1

# num_features = num_electrodes
# data_merge = np.delete(data_merge, rows_to_del, 0)
# # data_merge_windowless = np.delete(data_merge_windowless, rows_to_del, 0)


# labels_correlation_windowless = print_value_counts(labels_list_windowless)

# # Compute class weights
# class_weights = compute_class_weight(
#     class_weight="balanced", classes=np.unique(labels_list), y=labels_list
# )
# print(labels_correlation_windowless)

# # Convert class weights to a dictionary
# class_weights_dict = dict(enumerate(class_weights))

# # class_weights_dict[0] = class_weights_dict[0] * 1.5
# # class_weights_dict[3] = class_weights_dict[3] * 1.2
# print(class_weights_dict)

# train_ratio = 0.80
# validation_ratio = 0.20
# test_ratio = 0.20

# k = 5  # Number of folds
# kf = KFold(n_splits=k, shuffle=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     data_merge, labels, train_size=train_ratio, shuffle=True
# )

# # values = x_train.reshape(-1, x_train.shape[-1])
# # scaler = StandardScaler()
# # scaler = scaler.fit(values)
# # x_train_norm = scaler.transform(values).reshape(x_train.shape)
# # # Standardization Validation
# # # values_val = x_train.reshape(-1, x_train.shape[-1])
# # # x_val_norm = scaler.transform(values_val).reshape(x_train.shape)
# # # Standardization Test
# # values_test = x_test.reshape(-1, x_test.shape[-1])
# # x_test_norm = scaler.transform(values_test).reshape(x_test.shape)

# # correlate_function(
# #     num_features,
# #     4,
# #     data_merge_windowless,
# #     labels_correlation_windowless,
# #     f,
# #     window,
# #     correlation_scores,
# # )

# print("YO")
# print(len(labels_list_windowless))
# print(correlation_scores)
# print("YO")

# # x_train_less, x_test_less, y_train_less, y_test_less = train_test_split(
# #     data_merge_windowless.transpose(),
# #     labels_list_window,
# #     train_size=train_ratio,
# #     shuffle=True,
# #     stratify=labels_list_window,
# # )
# into = 0
# for train_index, val_index in kf.split(x_train):
#     into += 1
#     model = ENGNet2(
#         4,
#         num_features,
#     )
#     model_name = "ENGNet2"
#     model.summary()
#     # print("ciao")
#     # print(train_index)
#     # print(val_index)
#     # print("ciao")
#     x_train_k, x_val = x_train[train_index], x_train[val_index]
#     y_train_k, y_val = y_train[train_index], y_train[val_index]

#     # x_train, x_val, labels_train, labels_val = train_test_split(
#     #     x_train,
#     #     labels_train,
#     #     train_size=1 - validation_ratio,
#     #     random_state=420,
#     #     stratify=labels_train,
#     # )

#     # print(num_features)
#     # print(sfreq)
#     # model = ENGNet2(
#     #     4, num_features, x_train_norm.shape[0], window, class_weights_dict
#     # )
#     # model_name = "ENGNet2"

#     # model.summary()

#     layer_to_save_weights = model.layers[2]
#     weights_to_save = layer_to_save_weights.get_weights()

#     output_folder_cv = "../"

#     checkpoint_path = os.path.join(output_folder_cv, "best_model_checkpoint.h5")
#     model_checkpoint = tfk.callbacks.ModelCheckpoint(
#         checkpoint_path,
#         monitor="val_f1_score",
#         save_best_only=True,
#         save_weights_only=True,
#         verbose=1,
#     )

#     # Save the weights to a file
#     weights_file_path = str(into) + "saved_weights.h5"
#     with tf.keras.utils.CustomObjectScope(
#         {"GlorotUniform": tf.keras.initializers.GlorotUniform}
#     ):
#         layer_to_save_weights.set_weights(weights_to_save)
#         model.save_weights(weights_file_path)
#     print(x_train.shape)
#     print(y_train_k.shape)
#     history = model.fit(
#         x=x_train_k,
#         y=y_train_k,
#         epochs=5,
#         validation_data=(x_val, y_val),
#         callbacks=[
#             tfk.callbacks.EarlyStopping(
#                 monitor="val_f1_score",
#                 mode="max",
#                 patience=30,
#                 restore_best_weights=True,
#             ),
#             tfk.callbacks.ReduceLROnPlateau(
#                 monitor="val_f1_score", mode="max", patience=30, factor=0.5
#             ),
#             model_checkpoint,
#         ],
#     ).history

#     model.load_weights(checkpoint_path)

#     # Plotting, da aggiiungere la loss
#     best_epoch = np.argmax(history["val_f1_score"])
#     plt.figure(figsize=(17, 4))
#     plt.plot(history["loss"], label="Training loss", alpha=0.8, color="#ff7f0e")
#     plt.plot(history["val_loss"], label="Validation loss", alpha=0.9, color="#5a9aa5")
#     # plt.axvline(x=best_epoch, label="Best epoch", alpha=0.3, ls="--", color="#5a9aa5")
#     plt.title("Categorical Crossentropy")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     # plt.show()

#     plt.figure(figsize=(17, 4))
#     plt.plot(
#         history["val_f1_score"],
#         label="Training accuracy",
#         alpha=0.8,
#         color="#ff7f0e",
#     )
#     plt.plot(
#         history["val_f1_score"],
#         label="Validation accuracy",
#         alpha=0.9,
#         color="#5a9aa5",
#     )
#     # plt.axvline(x=best_epoch, label="Best epoch", alpha=0.3, ls="--", color="#5a9aa5")
#     plt.title("F1Score")
#     plt.legend()
#     plt.grid(alpha=0.3)

#     img_name = (
#         str(current_time)
#         + "_"
#         + str(num_features)
#         + "_"
#         + model_name
#         + "_"
#         + str(model.count_params())
#         + ".png"
#     )
#     file_path = os.path.join("img_results", img_name)

#     plt.savefig(file_path)

#     # plt.show()

#     plt.figure(figsize=(18, 3))
#     plt.plot(history["lr"], label="Learning Rate", alpha=0.8, color="#ff7f0e")
#     # plt.axvline(x=best_epoch, label="Best epoch", alpha=0.3, ls="--", color="#5a9aa5")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     # plt.show()

#     model.save("on_a_gang_model" + str(into))
#     prediction = model.predict(x_test)
#     cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(prediction, axis=-1))

#     # Compute the classification metrics
#     accuracy = accuracy_score(
#         np.argmax(y_test, axis=-1), np.argmax(prediction, axis=-1)
#     )
#     precision = precision_score(
#         np.argmax(y_test, axis=-1),
#         np.argmax(prediction, axis=-1),
#         average="macro",
#     )
#     recall = recall_score(
#         np.argmax(y_test, axis=-1),
#         np.argmax(prediction, axis=-1),
#         average="macro",
#     )
#     f1 = f1_score(
#         np.argmax(y_test, axis=-1),
#         np.argmax(prediction, axis=-1),
#         average="macro",
#     )
#     print("Accuracy:", accuracy.round(4))
#     print("Precision:", precision.round(4))
#     print("Recall:", recall.round(4))
#     print("F1:", f1.round(4))

#     # Plot the confusion matrix
#     plt.figure(figsize=(10, 8))
#     x_axis_labels = ["prop-", "prop+", "touch", "noci "]
#     y_axis_labels = ["prop-", "prop+", "touch", "noci "]
#     sns.heatmap(
#         cm.T,
#         annot=True,
#         cmap="Blues",
#         fmt="d",
#         annot_kws={"size": 12},
#         xticklabels=x_axis_labels,
#         yticklabels=y_axis_labels,
#     )
#     plt.xlabel("True labels")
#     plt.ylabel("Predicted labels")

#     img_name = (
#         str(current_time)
#         + "_"
#         + str(num_features)
#         + "_"
#         + model_name
#         + "_"
#         + str(model.count_params())
#         + "_conf"
#         + ".png"
#     )
#     file_path = os.path.join("img_results", img_name)

#     plt.savefig(file_path)

# model1 = tf.keras.models.load_model("on_a_gang_model" + "1")
# model2 = tf.keras.models.load_model("on_a_gang_model" + "2")
# model3 = tf.keras.models.load_model("on_a_gang_model" + "3")
# model4 = tf.keras.models.load_model("on_a_gang_model" + "4")
# model5 = tf.keras.models.load_model("on_a_gang_model" + "5")

# predictions1 = model1.predict(x_test)
# predictions2 = model2.predict(x_test)
# predictions3 = model3.predict(x_test)
# predictions4 = model4.predict(x_test)
# predictions5 = model5.predict(x_test)
# # predictions.shape

# # Assuming you have predictions from 5 models stored in a list
# model_predictions = [
#     predictions1,
#     predictions2,
#     predictions3,
#     predictions4,
#     predictions5,
# ]

# # Convert predictions to numpy arrays
# model_predictions_np = [np.array(pred) for pred in model_predictions]
# print(model_predictions_np)
# # Combine predictions by averaging
# combined_predictions = np.mean(model_predictions_np, axis=0)
# print(combined_predictions)

# # Take the class with the highest average prediction
# final_prediction = combined_predictions
# print(final_prediction)

# cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(final_prediction, axis=-1))

# # Compute the classification metrics
# accuracy = accuracy_score(
#     np.argmax(y_test, axis=-1), np.argmax(final_prediction, axis=-1)
# )
# precision = precision_score(
#     np.argmax(y_test, axis=-1),
#     np.argmax(final_prediction, axis=-1),
#     average="macro",
# )
# recall = recall_score(
#     np.argmax(y_test, axis=-1),
#     np.argmax(final_prediction, axis=-1),
#     average="macro",
# )
# f1 = f1_score(
#     np.argmax(y_test, axis=-1),
#     np.argmax(final_prediction, axis=-1),
#     average="macro",
# )
# print("Accuracy:", accuracy.round(4))
# print("Precision:", precision.round(4))
# print("Recall:", recall.round(4))
# print("F1:", f1.round(4))

# # Plot the confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm.T, annot=True, cmap="Blues", fmt="d", annot_kws={"size": 12})
# plt.xlabel("True labels")
# plt.ylabel("Predicted labels")

# img_name = (
#     str(current_time)
#     + "_"
#     + str(num_features)
#     + "_"
#     + model_name
#     + "_"
#     + str(model.count_params())
#     + "_conf"
#     + ".png"
# )
# file_path = os.path.join("img_results", img_name)

# plt.savefig(file_path)
# # plt.show()

# f.write(
#     str(current_time)
#     + "_"
#     + str(num_features)
#     + "_"
#     + model_name
#     + "_"
#     + str(model.count_params())
#     + " file"
#     + "\n"
# )
# for i in channel_bool:
#     f.write(str(i))

# f.write("\n")
# f.write(str(accuracy))
# f.write("\n")
# f.write(str(precision))
# f.write("\n")
# f.write(str(recall))
# f.write("\n")
# f.write(str(f1))

# f.write("\n")
# f.write("\n")
# f.write("\n")
# # f.close()
