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


def read_noci(counts, count, x, ult_data, raw, file, fil, tui):
    rangee = 1
    rangee_half = 1
    fs = 30000
    # Specificare la lunghezza della finestra mobile
    lunghezza_finestra = int(fs * 0.5)
    print("babbaba")
    print(len(x))
    x = raw.get_data()
    i = 17
    # Calcolare offset con la media mobile
    offset = moveavg(x[i], lunghezza_finestra)
    print(offset)

    # Tolgo offset
    a = np.subtract(x[i], offset)

    # soglia = 0.00000138

    # plt.figure()
    # plt.plot(xtot, a)
    # plt.title("sensor no offset")

    # Calcolare la mean absolute value (MAV) su una finestra mobile
    mav = moveavg(np.abs(x[i]), lunghezza_finestra)

    print(len(mav))
    # plt.show()

    plt.plot(x[0] * 100000)
    plt.plot(mav[0:1000000], color="orange")
    soglia = np.mean(mav)

    plt.hlines(y=soglia, xmin=0, xmax=1000000, linewidth=2, color="r")
    # plt.figure()
    # plt.plot(xtot, mav)
    # plt.title("mav")

    # Ora cerco on off con la soglia

    # Trovare i valori che superano la soglia
    j = 1
    contr = 1
    timeon = []
    timeoff = []

    for o in range(len(mav)):

        if mav[o] > soglia:
            if contr == 1:
                contr = 0
                # if len(timeon) == 0 or o - timeon[len(timeon) - 1] > 1000:
                timeon.append(o - 800)  # c'è un errore introdotto dalle medie
        else:
            if contr == 0:
                contr = 1
                # if (
                #     len(timeoff) == 0
                #     or o - timeoff[len(timeoff) - 1] > 1000
                #     and (o - timeon[len(timeon) - 1]) > 1000
                # ):
                timeoff.append(o + 400)
                # j += 1
    print("timeon")

    # del_on = []
    # del_off = []

    # for iv in del_on:
    #     timeon.remove(iv)
    # for ivo in del_off:
    #     timeoff.remove(ivo)
    # if len(timeoff) > 0:
    #     for itt in range(len(timeoff)):
    #         if timeoff[itt] - timeon[itt] < 200:
    #             del_on.append(timeon[itt])
    #             del_off.append(timeoff[itt])
    #     counto = 0

    #     # print(timeon)
    #     # print(timeoff)

    #     for itto in range(len(timeoff) - 1):
    #         if not (timeon[itto + 1] > timeoff[itto]):
    #             del timeon[itto + 1]

    for q in timeon:
        plt.axvline(x=q, color="g", linestyle="--")
    for w in timeoff:
        plt.axvline(x=w, color="r", linestyle="--")
    # plt.plot(h[i])
    # plt.show()
    print(timeon)
    print(timeoff)
    for ho in range(min(len(timeon), len(timeoff))):
        print("()_()")
        y = x[:16, timeon[ho] : timeoff[ho]]
        np.save(
            "numpy_arrays/"
            + "animal "
            + tui
            + "/"
            + fil
            + "/"
            + file
            + "__"
            + str(ho)
            + ".npy",
            y,
        )


def read_prop(counts, count, x, ult_data, yt, file, fil, tui, raw):
    # plt.plot(x[0])
    # plt.show()
    cutoff_time = 250  # ms
    cut_piece = cutoff_time * 30000 / 1000
    is_first_time_on = 1
    rangee = len(yt)
    rangee_half = int(len(yt) / 2)
    for i in range(rangee - 1):
        # if is_first_time_on:
        #     if i % 2 == 0:
        #         ult_data[counts, count, i] = (yt[i][0] + int(cut_piece * 2)) / 6
        #     elif i % 2 != 0:
        #         ult_data[counts, count, i] = (yt[i - 1][0] + int(cut_piece * 11)) / 6
        # else:
        #     if i % 2 != 0:
        #         ult_data[counts, count, i] = (yt[i][0] + int(cut_piece * 2)) / 6
        #     elif i % 2 == 0:
        #         ult_data[counts, count, i] = (yt[i - 1][0] + int(cut_piece * 11)) / 6
        if i % 2 == 0:
            ult_data[counts, count, i] = (yt[i + 1][0] + int(cut_piece * 1)) / 6
        elif i % 2 != 0:
            ult_data[counts, count, i] = (yt[i][0] + int(cut_piece * 11)) / 6
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
        print(int(ult_data[counts, count, j * 2]))
        print(int(ult_data[counts, count, (j * 2) + 1]))
        print(len(y[0]))
        current_time = datetime.now()
        np.save(
            "numpy_arrays/"
            + "animal "
            + tui
            + "/"
            + fil
            + "/"
            + file
            + "__"
            + str(j)
            + ".npy",
            y,
        )

    for i in range(16):
        plt.plot(x[i])
        h = raw.get_data()
        # plt.plot(h[16] / 50000, color="k")  # black
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


def read_touch(counts, count, x, ult_data, yt, file, fil, tui, raw):
    plt.plot(x[0])
    # plt.show()
    cutoff_time = 250  # ms
    cut_piece = cutoff_time * 30000 / 1000
    is_first_time_on = 0
    rangee = len(yt)
    rangee_half = int(len(yt) / 2)
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

    for k in ult_data[counts, count, :]:
        print(k)
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
            "numpy_arrays/"
            + "animal "
            + tui
            + "/"
            + fil
            + "/"
            + file
            + "__"
            + str(j)
            + ".npy",
            y,
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
        # plt.plot(x[i])
        h = raw.get_data()
        plt.plot(h[16] / 50000, color="k")  # black
        plt.show()
        # plt.savefig("numpy_arrays/" + "img__  " + tui + " " +
        #             file + "  channel  " + str(i) + "h.jpg")


def moveavg(arr, window_size):

    return np.convolve(arr, np.ones(window_size), "same") / window_size
