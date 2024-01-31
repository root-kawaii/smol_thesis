import os
from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

from datetime import datetime
from correlation import correlate_function
import pywt


tfk = tf.keras
tfkl = tf.keras.layers

# def eval(position):
animal_num = input("Enter animal number 1,2 or 3 \n")
animal_num = animal_num + "/"
patho = "../pso/data/animal "
folders = [f for f in os.listdir(patho + str(animal_num))]

# plt.switch_backend('TkAgg')

# files_nev = [f for f in os.listdir("../pso/data_nev")]
data_list = []
# other_list = []
labels_list = []
counter = 0


position = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]


for l in range(100):
    f = open("results.txt", "a")
    current_time = datetime.now()
    data_list = []
    labels_list = []
    for i in range(0, 16):
        position[i] = 1
        print('io')
    for counts, fil in enumerate(folders):
        f.write(str(counts) + " is " + fil + "\n")
        files = [f for f in os.listdir(patho + str(animal_num) + fil)]
        for count, file in enumerate(files):

            num_electrodes = 0
            # num_sensors = 0
            # their code #################
            # It's dealing with files that Elisa's old code generated, so we have to adequate it
            # sample=sample[0,:,0:16]
            # their code #################
            raw = mne.io.read_raw_nsx(
                patho + str(animal_num) + fil + "/" + file, preload=True)
            raw.describe()
            sfreq = 100
            # apply a highpass filter from 1 Hz upwards
            # replace baselining with high-pass
            # test divide by 4
            # raw = raw/4
            # raw._data *= 1e+3
            # raw.filter(l_freq=800, h_freq=None)
            raw.filter(l_freq=800, h_freq=2500)

            raw.resample(5000)
            raw.drop_channels(["flexsensor", "pressuresensor", "motors"])
            # raw.compute_psd(fmax=50).plot(
            #     picks="data", exclude="bads", amplitude=False)

            # ica = mne.preprocessing.ICA(
            #     n_components=4, random_state=97, max_iter=800)
            # ica.fit(raw)
            # raw.load_data()
            # ica.apply(raw)

            time_on = 0.5  # definiti dal sensore codice davide da vedere, cambiano da animale ad animale e da classe a classe
            time_off = 4  # definiti del sensore
            # finestra lunga 2.5 secondi circa nel dataset ma ogni tanto
            # da usare finestre di 100ms
            x = raw.get_data(tmin=time_on, tmax=time_off)
            # value_to_filter = 0.000030
            # x = x[:, :int(len(x[0])*0.01)]
            rows_to_del = []
            for j in range(len(x)):
                if (position[j] != 1 and j < 16):
                    rows_to_del.append(j)
                # if (j > 15):
                #     rows_to_del.append(j)
                elif (position[j] == 1 and j < 16):
                    num_electrodes += 1
                # elif (j > 15):
                    # num_sensors += 1
            num_features = num_electrodes
            print(len(x[1]))
            x = np.delete(x, rows_to_del, 0)

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

            print(counter)
            # for t in range(len(x)):
            #     for y in range(len(x[t])):
            # print(x[t, y])

            values = x.reshape(-1, x.shape[-1])
            scaler = StandardScaler()
            scaler = scaler.fit(values)
            x = scaler.transform(values).reshape(x.shape)

            # for t in range(len(x)):
            #     for y in range(len(x[t])):
            #         if (abs(x[t, y]) > 0.0000030):
            #             x[t, y] = 0

            print('cleaned')

            # create a numpy array of EEG data from the MNE raw object
            # eeg_array = x[:, 0:len(x[0])]

            cic = int(len(x[0]) / sfreq)

            # extract the sampling frequency from the MNE raw object

            # (optional) make sure your asr is only fitted to clean parts of the data
            # pre_cleaned, _ = clean_windows(x, sfreq, max_bad_chans=0.1)
            # fit the asr
            # M, T = asr_calibrate(x, sfreq, cutoff=15)
            # # apply it
            # clean_array = asr_process(x, sfreq, M, T)

            widths = np.arange(1, (sfreq*2)+1)
            datas = np.zeros(
                (cic, (sfreq*2), sfreq, num_features), dtype=float)
            for i in range(num_features):
                for j in range(cic):
                    datas[j, :, :, i], freq = pywt.cwt(x[i, (sfreq*j): sfreq*(j+1)],
                                                       widths, 'mexh', sampling_period=1/5000)
            print(datas.shape)
            data_list.append(datas)
            labels_list.extend([counts] * (cic))

    # here we call correlation
    # correlate_function(num_features, 10, other_list, f)
    x = []
    data_merge = np.concatenate(data_list, axis=0)
    data_list = []
    data = np.empty
    # # data_merge = np.expand_dims(data_merge, axis=2)

    labels = np.array(labels_list)
    # print(labels)
    labels2 = tfk.utils.to_categorical(labels_list)
    print(labels2.shape)

    train_ratio = 0.60
    validation_ratio = 0.20
    test_ratio = 0.20

    x_train, x_test, labels_train, labels_test = train_test_split(
        data_merge, labels2, train_size=train_ratio)

    x_test, x_val, y_test, y_val = train_test_split(x_test, labels_test, test_size=test_ratio/(
        validation_ratio + test_ratio), random_state=420, stratify=labels_test)

    # values = x_train.reshape(-1, x_train.shape[-1])
    # scaler = StandardScaler()
    # scaler = scaler.fit(values)
    # x_train_norm = scaler.transform(values).reshape(x_train.shape)
    # # Standardization Validation
    # values_val = x_val.reshape(-1, x_val.shape[-1])
    # x_val_norm = scaler.transform(values_val).reshape(x_val.shape)
    # # Standardization Test
    # values_test = x_test.reshape(-1, x_test.shape[-1])
    # x_test_norm = scaler.transform(values_test).reshape(x_test.shape)

    print(num_features)
    print(sfreq)
    model = ENGNet2(4, num_features, sfreq, sfreq*2)
    model_name = "ENGNet2"

    model.summary()

    # layer_to_save_weights = model.layers[2]
    # weights_to_save = layer_to_save_weights.get_weights()

    # # Save the weights to a file
    # weights_file_path = "saved_weights.h5"
    # with tf.keras.utils.CustomObjectScope({'GlorotUniform': tf.keras.initializers.GlorotUniform}):
    #     layer_to_save_weights.set_weights(weights_to_save)
    #     model.save_weights(weights_file_path)

    history = model.fit(
        x=x_train,
        y=labels_train,
        batch_size=200,
        epochs=500,
        validation_data=(x_val, y_val),
        callbacks=[
            tfk.callbacks.EarlyStopping(
                monitor='val_f1_score', mode='max', patience=10, restore_best_weights=True),
            tfk.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score', mode='max', patience=10, factor=0.5, min_lr=1e-5)
        ]
    ).history

    # Plotting, da aggiiungere la loss
    best_epoch = np.argmax(history['val_f1_score'])
    plt.figure(figsize=(17, 4))
    plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='Validation loss',
             alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch',
                alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Categorical Crossentropy')
    plt.legend()
    plt.grid(alpha=.3)
    # plt.show()

    plt.figure(figsize=(17, 4))
    plt.plot(history['f1_score'], label='Training accuracy',
             alpha=.8, color='#ff7f0e')
    plt.plot(history['val_f1_score'], label='Validation accuracy',
             alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch',
                alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(alpha=.3)

    img_name = str(current_time) + "_" + str(num_features) + \
        "_" + model_name + "_" + str(model.count_params()) + '.png'
    file_path = os.path.join("img_results", img_name)

    plt.savefig(file_path)

    # plt.show()

    plt.figure(figsize=(18, 3))
    plt.plot(history['lr'], label='Learning Rate', alpha=.8, color='#ff7f0e')
    plt.axvline(x=best_epoch, label='Best epoch',
                alpha=.3, ls='--', color='#5a9aa5')
    plt.legend()
    plt.grid(alpha=.3)
    # plt.show()

    model.save('on_a_gang_model')

    predictions = model.predict(x_test)
    predictions.shape

    cm = confusion_matrix(np.argmax(y_test, axis=-1),
                          np.argmax(predictions, axis=-1))

    # Compute the classification metrics
    accuracy = accuracy_score(np.argmax(y_test, axis=-1),
                              np.argmax(predictions, axis=-1))
    precision = precision_score(
        np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    recall = recall_score(np.argmax(y_test, axis=-1),
                          np.argmax(predictions, axis=-1), average='macro')
    f1 = f1_score(np.argmax(y_test, axis=-1),
                  np.argmax(predictions, axis=-1), average='macro')
    print('Accuracy:', accuracy.round(4))
    print('Precision:', precision.round(4))
    print('Recall:', recall.round(4))
    print('F1:', f1.round(4))

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.T, cmap='Blues')
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')

    img_name = str(current_time) + "_" + str(num_features) + \
        "_" + model_name + "_" + str(model.count_params()) + '_conf' + '.png'
    file_path = os.path.join("img_results", img_name)

    plt.savefig(file_path)
    # plt.show()

    f.write(str(current_time) + "_" + str(num_features) +
            "_" + model_name + "_" + str(model.count_params()) + ' file' + "\n")
    for i in position:
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
