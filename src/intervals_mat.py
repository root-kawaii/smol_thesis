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
ult_data = np.zeros(
    (shape_of_date_nev))

for counts, fil in enumerate(folders_nev):
    if (fil != ".DS_Store"):
        files = [f[:-4] for f in os.listdir(animal_num + fil)]
        print(files[0])
        for count, file in enumerate(files):
            if (file != ".DS_Store"):
                data_times = []
                num_electrodes = 0
                # num_sensors = 0
                # their code #################
                # It's dealing with files that Elisa's old code generated, so we have to adequate it
                # sample=sample[0,:,0:16]
                # their code #################
                raw = mne.io.read_raw_nsx(
                    "data/" + "animal " + tui + "/" + fil + "/" + file + ".ns5", preload=True)
                raw.describe()
                sfreq = 100
                # apply a highpass filter from 1 Hz upwards
                # replace baselining with high-pass
                # test divide by 4
                # raw = raw/4
                # raw._data *= 1e+3
                # raw.filter(l_freq=800, h_freq=None)
                # raw.filter(l_freq=800, h_freq=2500)

                # raw.resample(5000)
                raw.drop_channels(
                    ["flexsensor", "pressuresensor", "motors"])
                x = raw.get_data()

                print(animal_num + fil + "/" + file)
                r = neo.io.BlackrockIO(
                    filename=animal_num + fil + "/" + file + ".nev")
                bl = r.read_block(lazy=False)
                bl = r.nev_data
                th = bl["Comments"]
                yt = th[0]
                for i in range(min(len(yt)-1, shape_of_date_nev[2])):
                    ult_data[counts, count, i] = yt[i][0]
                for i in range(len(x)):
                    for j in range(min(int((len(yt)-1)/2), int(shape_of_date_nev[2]/2))):
                        y = x[i, int(ult_data[counts, count, j*2]):
                              int(ult_data[counts, count, (j*2)+1])]
                current_time = datetime.now()
                print(len(y))
                np.save(str(current_time) + "h.npy", y)

print(ult_data.shape)
