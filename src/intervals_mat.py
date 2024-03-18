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


import asrpy
from asrpy import asr_calibrate, asr_process, clean_windows

from datetime import datetime
from read_files_functions import *

tui = input("Enter animal number 1,2 or 3 \n")
animal_num = tui + "/"
patho = "../src/data/animal "
patho_nev = "../src/data_nev/animal "
folders = [f for f in os.listdir(patho + str(animal_num))]
folders_nev = [f for f in os.listdir(patho_nev + str(animal_num))]

path_to_animal_folder = patho_nev + str(animal_num)
shape_of_date_nev = (4, 10, 99)

# def extract_intervals(path_to_animal_folder, shape_of_date_nev) -> np.ndarray:
animal_num = path_to_animal_folder
folders_nev = [f for f in os.listdir(animal_num)]
ult_data = np.zeros((shape_of_date_nev))

for counts, fil in enumerate(folders_nev):
    if fil != ".DS_Store":
        files = [f[:-4] for f in os.listdir(animal_num + fil)]
        for count, file in enumerate(files):
            if file != ".DS_Store":
                # their code #################
                # It's dealing with files that Elisa's old code generated, so we have to adequate it
                # sample=sample[0,:,0:16]
                # their code #################
                raw = mne.io.read_raw_nsx(
                    "data/" + "animal " + tui + "/" + fil + "/" + file + ".ns5",
                    preload=True,
                )
                raw.filter(l_freq=800, h_freq=2450)
                raw.describe()
                # plt.plot(x[0])
                # plt.show()
                raw = raw.resample(5000)

                x = raw.get_data()
                print(len(x))
                rows_to_del = []
                rows_to_del.append(16)
                rows_to_del.append(17)
                if len(x) == 19:
                    rows_to_del.append(18)
                x = np.delete(x, rows_to_del, 0)
                print(len(x))
                for t in range(len(x)):
                    for y in range(len(x[t])):
                        # print(x[t, y])
                        # x[t, y] = int(x[t, y] / 4)
                        if abs(x[t, y]) > 0.000030:
                            # print(x[t, y])
                            x[t, y] = 0.000030
                            # print("yoyo")
                # plt.plot(x[0], color="orange")
                # plt.show()

                # M, T = asr_calibrate(x, 5000, cutoff=15)
                # # apply it
                # clean_temp = asr_process(x, 5000, M, T)

                print(animal_num + fil + "/" + file)
                r = neo.io.BlackrockIO(filename=animal_num + fil + "/" + file + ".nev")
                bl = r.read_block(lazy=False)
                bl = r.nev_data
                th = bl["Comments"]
                yt = th[0]

                # data elements for each ms * 250ms
                if fil == "noci":
                    read_noci(
                        counts,
                        count,
                        x,
                        ult_data,
                        raw,
                        file,
                        fil,
                        tui,
                    )

                elif fil == "touch":
                    read_touch(
                        counts,
                        count,
                        x,
                        ult_data,
                        yt,
                        file,
                        fil,
                        tui,
                        raw,
                    )
                else:
                    read_prop(
                        counts,
                        count,
                        x,
                        ult_data,
                        yt,
                        file,
                        fil,
                        tui,
                        raw,
                    )
