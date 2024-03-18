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


def get_label_from_path(name):
    # Remove file extension
    label = name.replace(".mat", "")
    # Remove intial part of file name, eg "sample_0161_"
    label = label[12:]
    return label


# Encodes label depending on the name of the file
def encod_label(lab):
    if lab[0:4] == "noci" or lab[0:4] == "Noci" or lab[0:5] == "Pinch":
        label = 0
    elif lab[0:4] == "prop" and lab[11] == "-":
        label = 1
    elif lab[0:4] == "prop":
        label = 2
    elif lab[0:5] == "touch":
        label = 3
    return label


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

check = 0
f = open("results.txt", "a")
current_time = datetime.now()
for i in range(0, 16):
    channel_bool[i] = 1
    print("io")
for i, m in enumerate(files_mat):
    if i % 100 != 0:
        mat = scipy.io.loadmat("../100ms/" + m)
        data_mat = mat["to_save"]
        labo = get_label_from_path(m)
        laboo = encod_label(labo)
        all_classes.append(data_mat.transpose())
        labels_list.extend([laboo] * 1)
        labels_list_windowless.extend([laboo] * len(data_mat[0]))
    # if "_prop_angle_-30" in i or "_prop_angle_-20" in i or "_prop_angle_-10" in i:
    #     labels_list.extend([0] * 1)
    #     labels_list_windowless.extend([0] * len(data_mat[0]))
    # elif "noci_trial" in i:
    #     labels_list.extend([2] * 1)
    #     labels_list_windowless.extend([2] * len(data_mat[0]))
    # elif "_touch_" in i:
    #     labels_list.extend([3] * 1)
    #     labels_list_windowless.extend([3] * len(data_mat[0]))
    # else:
    #     labels_list.extend([1] * 1)
    #     labels_list_windowless.extend([1] * len(data_mat[0]))
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
print(data_merge.shape)

# data_merge = data_merge.transpose()
print(data_merge.shape)
data_merge_windowless = np.concatenate(data_merge, axis=0)
data_merge_windowless = data_merge_windowless.transpose()
print(data_merge_windowless.shape)
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
kf = KFold(n_splits=k, shuffle=True, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(
    data_merge, labels, train_size=train_ratio, shuffle=True, random_state=42
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
        epochs=50,
        validation_data=(x_val, y_val),
        # class_weight=class_weights_dict,
        callbacks=[
            tfk.callbacks.EarlyStopping(
                monitor="val_f1_score",
                mode="max",
                patience=15,
                restore_best_weights=True,
            ),
            tfk.callbacks.ReduceLROnPlateau(
                monitor="val_f1_score", mode="max", patience=15, factor=0.5
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
