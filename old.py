

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
labels = np.chararray((len(files)*5000))

position = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# def evaluateSol(position):

for count, file in enumerate(files):
    nsx_filepath = file

    r = io.BlackrockIO(
        filename="../pso/data/" + file)
    bl = r.read_block(lazy=False)
    seg = bl.segments[0]
    # transpose cause channels and inputs were swapped
    # select [0] cause in [1] we find flexsensor pressuresensor motors
    x = seg.analogsignals[0].transpose()
    # number of channels and number of points
    n_chan, n_pnts, = len(x), len(x[0])
    # ch_names = list()
    # create empty data with right sizes
    data = np.zeros((n_chan, 5000), dtype=float)
    for i in range(n_chan):
        # if our proposed position has 1 it means the corrisponding feature is enabled,
        #  otherwise we don't want that channel
        if (position[i] == 1):
            data[i, 0:5000] = x[i][0:5000]
        else:
            data[i, 0:5000] = None

    sfreq = int(seg.analogsignals[0].sampling_rate.magnitude)
    # kinda useless, i just write the number of electrode to comply with previous code and fill the create_info function
    ch_names = ['0', '1', '2', '3', '4', '5', '6', '7',
                '8', '9', '10', '11', '12', '13', '14', '15']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
    # create array with mne library
    raw = mne.io.RawArray(data, info)
    # print('/n/n')
    # print(raw)
    data_list.append(data)
    raw_list.append(raw)
    s = [count] * 5000
    print(s)
    labels[count*5000:((count+1)*5000)] = s
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
    if (i < len(data_list)-1):
        data_list[0] = np.append(data_list[0], data_list[i+1])

data_merge = np.zeros((n_chan, 5000*len(files)), dtype=float)
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
test = data_merge.transpose()[25000 + 1:]

labels_train = labels[0:25000]
labels_test = labels[25000+1:]


clf = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto'))
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
