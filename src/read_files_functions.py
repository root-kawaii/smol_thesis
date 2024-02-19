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


def read_noci(shape_of_date_nev, counts, count, x, ult_data, raw):
    rangee = 1
    rangee_half = 1
    fs = 30000
    # Specificare la lunghezza della finestra mobile
    lunghezza_finestra = int(fs * 0.5)
    print("babbaba")
    for i in range(len(x)):
        # plt.plot(x[i])
        # h = raw.get_data()
        # plt.plot(h[17] / 50000, color="k")  # black
        # plt.show()
        # Calcolare offset con la media mobile
        offset = moveavg(x[i], lunghezza_finestra)
        print(offset)

        # Tolgo offset
        a = np.subtract(x[i], offset)

        # plt.figure()
        # plt.plot(xtot, a)
        # plt.title("sensor no offset")

        # Calcolare la mean absolute value (MAV) su una finestra mobile
        mav = moveavg(np.abs(a), lunghezza_finestra)
        # print(len(mav))
        # plt.plot(mav[0:100000])
        # plt.show()

        # plt.figure()
        # plt.plot(xtot, mav)
        # plt.title("mav")

        # Ora cerco on off con la soglia
        soglia = 0.0000005

        # Trovare i valori che superano la soglia
        j = 0
        contr = 1
        timeon = []
        timeoff = []

        for i in range(len(mav)):

            if mav[i] > soglia:
                if contr == 1:
                    contr = 0
                    timeon.append(i + 8000)  # c'è un errore introdotto dalle medie
            else:
                if contr == 0:
                    contr = 1
                    timeoff.append(i - 8000)
                    j += 1
        print(timeon)
        print(timeoff)


def read_prop(shape_of_date_nev, counts, count, x, ult_data, yt, file, fil, tui, raw):
    cutoff_time = 250  # ms
    cut_piece = cutoff_time * 30000 / 1000
    is_first_time_on = 1
    rangee = shape_of_date_nev[2]
    rangee_half = int(shape_of_date_nev[2] / 2)
    for i in range(rangee):
        if is_first_time_on:
            if i % 2 == 0:
                ult_data[counts, count, i] = (yt[i][0] + int(cut_piece * 2)) / 6
            elif i % 2 != 0:
                ult_data[counts, count, i] = (yt[i - 1][0] + int(cut_piece * 11)) / 6
        else:
            if i % 2 != 0:
                ult_data[counts, count, i] = (yt[i][0] + int(cut_piece * 2)) / 6
            elif i % 2 == 0:
                ult_data[counts, count, i] = (yt[i - 1][0] + int(cut_piece * 11)) / 6
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
        plt.plot(x[i])
        h = raw.get_data()
        plt.plot(h[16] / 50000, color="k")  # black
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


def moveavg(arr, window_size):

    return np.convolve(arr, np.ones(window_size), "same") / window_size
