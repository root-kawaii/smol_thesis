import os
import neo
from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from sklearn.svm import SVC
import mne

# from mne.decoding import CSP

from datetime import datetime


def read_noci(shape_of_date_nev, counts, count, x, ult_data):
    rangee = 1
    rangee_half = 1

    # altro
    y = x[:, int(ult_data[counts, count, 0]) : int(ult_data[counts, count, 1])]


def read_prop(shape_of_date_nev, counts, count, x, ult_data, yt, file, fil, tui, raw):
    cutoff_time = 250  # ms
    cut_piece = cutoff_time * 30000 / 1000
    is_first_time_on = 1
    rangee = shape_of_date_nev[2]
    rangee_half = int(shape_of_date_nev[2] / 2)
    for i in range(rangee):
        if is_first_time_on:
            if i % 2 == 0:
                ult_data[counts, count, i] = (yt[i][0] + int(cut_piece)) / 6
            elif i % 2 != 0:
                ult_data[counts, count, i] = (yt[i][0] - int(cut_piece)) / 6
        else:
            if i % 2 != 0:
                ult_data[counts, count, i] = (yt[i][0] + int(cut_piece)) / 6
            elif i % 2 == 0:
                ult_data[counts, count, i] = (yt[i][0] - int(cut_piece)) / 6

    for i in range(0, len(ult_data[counts, count, :]) - 1, 2):
        print(ult_data[counts, count, i + 1] - ult_data[counts, count, i])
    for j in range(rangee_half):
        # if (fil == "touch"):
        y = x[
            :,
            int(ult_data[counts, count, (j * 2)]) : int(
                ult_data[counts, count, (j * 2) + 1]
            ),
        ]
        # else:
        #     y = x[:, int(ult_data[counts, count, (j*2)]):
        #           int(ult_data[counts, count, (j*2)+1])]
        # print(int(ult_data[counts, count, j*2]))
        # print(int(ult_data[counts, count, (j*2)+1]))
        print(len(y[0]))
        current_time = datetime.now()
        np.save(
            "numpy_arrays/" + tui + "/" + fil + "/" + file + "__" + str(j) + ".npy", y
        )

    for i in range(16):
        for h, val in enumerate(ult_data[counts, count]):
            if is_first_time_on:
                if h % 2 == 0:
                    # red timeoff green timeon
                    plt.axvline(x=val, color="g", linestyle="--")
                else:
                    plt.axvline(x=val, color="r", linestyle="--")
            else:
                if h % 2 == 0:
                    # red timeoff green timeon
                    plt.axvline(x=val, color="r", linestyle="--")
                else:
                    plt.axvline(x=val, color="g", linestyle="--")
        plt.plot(x[i])
        h = raw.get_data()
        plt.plot(h[16] / 50000, color="k")  # black
        plt.show()
        # plt.savefig("numpy_arrays/" + "img__  " + tui + " " +
        #             file + "  channel  " + str(i) + "h.jpg")


def read_touch(shape_of_date_nev, counts, count, x, ult_data, yt, file, fil, tui, raw):
    cutoff_time = 250  # ms
    cut_piece = cutoff_time * 30000 / 1000
    is_first_time_on = 0
    rangee = shape_of_date_nev[2]
    rangee_half = int(shape_of_date_nev[2] / 2)
    for i in range(rangee):
        if is_first_time_on:
            if i % 2 == 0:
                ult_data[counts, count, i] = (yt[i][0] + int(cut_piece)) / 6
            elif i % 2 != 0:
                ult_data[counts, count, i] = (yt[i - 1][0] + int(cut_piece * 11)) / 6
        else:
            if i % 2 != 0:
                ult_data[counts, count, i] = (yt[i][0] + int(cut_piece)) / 6
            elif i % 2 == 0:
                ult_data[counts, count, i] = (yt[i - 1][0] + int(cut_piece * 11)) / 6

    for i in ult_data[counts, count, :]:
        print(i)
    for j in range(rangee_half):
        # if (fil == "touch"):
        y = x[
            :,
            int(ult_data[counts, count, (j * 2)]) : int(
                ult_data[counts, count, (j * 2) + 1]
            ),
        ]
        # else:
        #     y = x[:, int(ult_data[counts, count, (j*2)]):
        #           int(ult_data[counts, count, (j*2)+1])]
        # print(int(ult_data[counts, count, j*2]))
        # print(int(ult_data[counts, count, (j*2)+1]))
        print(len(y[0]))
        current_time = datetime.now()
        np.save(
            "numpy_arrays/" + tui + "/" + fil + "/" + file + "__" + str(j) + ".npy", y
        )

    for i in range(16):
        for h, val in enumerate(ult_data[counts, count]):
            if is_first_time_on:
                if h % 2 == 0:
                    # red timeoff green timeon
                    plt.axvline(x=val, color="g", linestyle="--")
                else:
                    plt.axvline(x=val, color="r", linestyle="--")
            else:
                if h % 2 == 0:
                    # red timeoff green timeon
                    plt.axvline(x=val, color="r", linestyle="--")
                else:
                    plt.axvline(x=val, color="g", linestyle="--")
        plt.plot(x[i])
        h = raw.get_data()
        plt.plot(h[16] / 50000, color="k")  # black
        plt.show()
        # plt.savefig("numpy_arrays/" + "img__  " + tui + " " +
        #             file + "  channel  " + str(i) + "h.jpg")
