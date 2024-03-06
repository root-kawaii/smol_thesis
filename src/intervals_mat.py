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
data_times = []
data = []
animal_num = path_to_animal_folder
folders_nev = [f for f in os.listdir(animal_num)]
ult_data = np.zeros((shape_of_date_nev))

for counts, fil in enumerate(folders_nev):
    if fil != ".DS_Store":
        files = [f[:-4] for f in os.listdir(animal_num + fil)]
        for count, file in enumerate(files):
            if file != ".DS_Store":
                data_times = []
                num_electrodes = 0
                # their code #################
                # It's dealing with files that Elisa's old code generated, so we have to adequate it
                # sample=sample[0,:,0:16]
                # their code #################
                raw = mne.io.read_raw_nsx(
                    "data/" + "animal " + tui + "/" + fil + "/" + file + ".ns5",
                    preload=True,
                )
                raw.describe()
                raw.resample(5000)
                sfreq = 100
                frequo = 5000

                x = raw.get_data()
                print(len(x))
                rows_to_del = []
                rows_to_del.append(16)
                rows_to_del.append(17)
                if len(x) == 19:
                    rows_to_del.append(18)
                x = np.delete(x, rows_to_del, 0)
                print(len(x))

                print(animal_num + fil + "/" + file)
                r = neo.io.BlackrockIO(filename=animal_num + fil + "/" + file + ".nev")
                bl = r.read_block(lazy=False)
                bl = r.nev_data
                th = bl["Comments"]
                yt = th[0]
                print("oooooooooooo")
                print(len(yt))
                print("oooooooooooo")
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
