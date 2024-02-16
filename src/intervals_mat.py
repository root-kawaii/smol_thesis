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
    if (fil == "touch"):
        files = [f[:-4] for f in os.listdir(animal_num + fil)]
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
                raw.resample(5000)
                sfreq = 100
                frequo = 5000
                # apply a highpass filter from 1 Hz upwards
                # replace baselining with high-pass
                # test divide by 4
                # raw = raw/4
                # raw._data *= 1e+3
                # raw.filter(l_freq=800, h_freq=None)
                # raw.filter(l_freq=800, h_freq=2500)

                # raw.resample(5000)
                # raw.drop_channels(
                #     ["flexsensor", "pressuresensor", "motors"])
                x = raw.get_data()
                print(len(x))
                rows_to_del = []
                rows_to_del.append(16)
                rows_to_del.append(17)
                if (len(x) == 19):
                    rows_to_del.append(18)
                x = np.delete(x, rows_to_del, 0)
                print(len(x))
                # if (fil != "noci"):
                #     for t in range(len(x)):
                #         for y in range(len(x[t])):
                #             # x[t, y] = int(x[t, y] / 4)
                #             if (abs(x[t, y]) > 0.0000030):
                #                 # counter += 1
                #                 x[t, y] = 0.0000030

                print(animal_num + fil + "/" + file)
                r = neo.io.BlackrockIO(
                    filename=animal_num + fil + "/" + file + ".nev")
                bl = r.read_block(lazy=False)
                bl = r.nev_data
                th = bl["Comments"]
                yt = th[0]
                print(yt)
                cutoff_time = 250  # ms
                cut_piece = cutoff_time * 30000/1000  # data elements for each ms * 250ms
                if (fil == "noci"):
                    rangee = 1
                    rangee_half = 1

                    # altro
                    y = x[:, int(ult_data[counts, count, 0]):
                          int(ult_data[counts, count, 1])]

                else:
                    rangee = shape_of_date_nev[2]
                    rangee_half = int(shape_of_date_nev[2]/2)
                for i in range(rangee):
                    if (fil == "touch"):
                        if (i % 2 == 0):
                            ult_data[counts, count, i] = (
                                yt[i][0] + int(cut_piece))/6
                        elif (i % 2 != 0):
                            ult_data[counts, count, i] = (
                                yt[i-1][0] + int(cut_piece*10))/6
                    else:
                        if (i % 2 == 0):
                            ult_data[counts, count, i] = (
                                yt[i][0] + int(cut_piece))/6
                        elif (i % 2 != 0):
                            ult_data[counts, count, i] = (
                                yt[i][0] - int(cut_piece))/6
                for i in ult_data[counts, count, :]:
                    print(i)
                for j in range(rangee_half):
                    # if (fil == "touch"):
                    y = x[:, int(ult_data[counts, count, (j*2)]):
                          int(ult_data[counts, count, (j*2)+1])]
                    # else:
                    #     y = x[:, int(ult_data[counts, count, (j*2)]):
                    #           int(ult_data[counts, count, (j*2)+1])]
                    # print(int(ult_data[counts, count, j*2]))
                    # print(int(ult_data[counts, count, (j*2)+1]))
                    print(len(y[0]))
                    current_time = datetime.now()
                    np.save("numpy_arrays/" + tui + "/" +
                            fil + "/" + file + "__" + str(j) + ".npy", y)

                for i in range(16):
                    for h, val in enumerate(ult_data[counts, count]):
                        if (h % 2 == 0):
                            # red timeoff green timeon
                            plt.axvline(x=val, color='r', linestyle='--')
                        else:
                            plt.axvline(x=val, color='g', linestyle='--')
                    plt.plot(x[i])
                    h = raw.get_data()
                    plt.plot(h[16]/50000, color='k')  # black
                    plt.show()
                    # plt.savefig("numpy_arrays/" + "img__  " + tui + " " +
                    #             file + "  channel  " + str(i) + "h.jpg")

print(ult_data.shape)
