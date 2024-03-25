import neo
import os
from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
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
from correlation import (
    correlate_function_2,
    correlate_function,
    correlate_function_right,
)
import pywt

from builtins import range


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
        label = 3
    elif lab[0:4] == "prop" and lab[11] == "-":
        label = 0
    elif lab[0:4] == "prop":
        label = 1
    elif lab[0:5] == "touch":
        label = 2
    return label


def import_sample(path):
    mat = scipy.io.loadmat(path, mat_dtype=True)
    x = np.array(mat["to_save"])
    # file_idx=int(mat['file_idx'])
    # sample_idx=int(mat['sample_idx'])
    # original_sample=int(mat['original_sample'])
    # n_samples_original=int(mat['n_samples_original'])
    return x  # ,file_idx,sample_idx,original_sample,n_samples_original


def transform_data_not_transpose(file_name, file_paths, n_features):
    y_samp = np.empty(len(file_name))
    # file_idx_vec=np.empty(len(file_name))
    # sample_idx_vec=np.empty(len(file_name))
    # original_sample_vec=np.empty(len(file_name))
    # n_samples_original_vec=np.empty(len(file_name))
    sample = []
    for file_number in range(len(file_name)):

        # file_idx,sample_idx,original_sample,n_sample_original

        # print(file_number)
        sample = import_sample(file_paths[file_number])

        if len(np.shape(sample)) == 3:
            # It's dealing with files that Elisa's old code generated, so we have to adequate it
            sample = sample[0, :, 0:16]
        else:
            # Dealing with files that Elisa's new code, changed by Rafael, generated
            sample = np.transpose(sample)
        if file_number == 0:
            x_samp = np.empty((len(file_name), np.shape(sample)[0], n_features))

        # print(file_name[file_number])
        # Gets the label from the file name
        lab = get_label_from_path(file_name[file_number])
        # print(lab)
        # Stores the sample
        x_samp[file_number, :, :] = sample

        # Encodes the label to a value that will be the target
        y_samp[file_number] = encod_label(lab)

        # Encodes if the sample was obtained via overlapping or not and the characteristics of it so
        # The training set can be built with no overlapping regarding the test set
        # file_idx_vec[file_number]=file_idx
        # sample_idx_vec[file_number]=sample_idx
        # original_sample_vec[file_number]=original_sample
        # n_samples_original_vec[file_number]=n_samples_original

        # x_samp_trasposta = np.transpose(x_samp, (0, 2, 1))

        # , file_idx_vec, sample_idx_vec, original_sample_vec, n_samples_original_vec

    return x_samp, y_samp


files_mat = [f for f in os.listdir("../100ms/")]

# Initialize variables to store cross-validation results
cv_results = {
    "val_accuracy": [],
    "val_f1_score": [],
    "val_weighted_f1_score": [],
    "val_best_weights": [],
    "val_loss": [],
    "train_acc_history": [],
    "val_acc_history": [],
    "val_confusion_matrix": [],
}
test_results = {
    "test_accuracy": [],
    "test_f1_score": [],
    "test_weighted_f1_score": [],
    "test_confusion_matrix": [],
}

data_list = []
labels_list = []
labels_list_windowless = []
counter = 0
all_classes = []
all_classes_windowless = []
correlation_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
channel_bool = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
num_features = 0
for i in channel_bool:
    if i:
        num_features += 1

num_electrodes = 0
check = 0
f = open("results.txt", "a")
current_time = datetime.now()

path_folder = "../100ms/"
file_name = [f for f in os.listdir("../100ms/")]
file_paths = []
for file_number in range(len(file_name)):
    file = os.path.join(path_folder, file_name[file_number])
    file_paths.append(file)


x_samp, y_samp = transform_data_not_transpose(file_name, file_paths, 16)


print(x_samp.shape)


train_ratio = 0.80
validation_ratio = 0.20
test_ratio = 0.20

k = 5  # Number of folds
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

x_samp_2 = np.concatenate(x_samp, axis=0)

x_samp_3 = np.transpose(x_samp_2, (1, 0))
# for i, j in enumerate(x_samp):
#     x_samp_3[i] = j.transpose()
print(x_samp.shape)
x_samp = np.transpose(x_samp)
print(x_samp.shape)

rows_to_del = []
for j in range(len(channel_bool)):
    if channel_bool[j] != 1 and j < 16:
        rows_to_del.append(j)
    elif channel_bool[j] == 1 and j < 16:
        num_electrodes += 1

x_samp_4 = np.delete(x_samp_3, rows_to_del, 0)
print(x_samp_4.shape)
x_samp = np.delete(x_samp, rows_to_del, 0)

print(x_samp.shape)
x_samp_t = np.transpose(x_samp)
x_samp_tt = np.transpose(x_samp_t, (0, 2, 1))
print(x_samp_tt.shape)


new = scipy.signal.correlate(x_samp_tt[11][2], x_samp_tt[10][2])
plt.plot(new)
plt.show()


x_train, x_test, y_train, y_test = train_test_split(
    x_samp_tt, y_samp, train_size=train_ratio, shuffle=True, random_state=42
)

y_samp_2 = []
print(y_samp.shape)
for i, item in enumerate(y_samp):
    y_samp_2.extend([item] * 500)
y_samp_2 = np.array(y_samp_2)

print(y_samp_2.shape)

x_samp_4 = x_samp_4.transpose()

labels_correlation_windowless = print_value_counts(y_samp_2)

voting = {}
for j in range(17):
    voting[j] = 0
print(voting)

# for train_index, val_index in kf.split(x_train, y_train):

#     x_train_k, x_val = (
#         x_train[train_index],
#         x_train[val_index],
#     )
#     y_train_k, y_val = y_train[train_index], y_train[val_index]
#     # labels_correlation_windowless = print_value_counts(y_val)

#     print(x_val.shape)

#     correlate_function_right(
#         num_features,
#         4,
#         x_val,
#         y_val,
#         # labels_correlation_windowless,
#         f,
#         500,
#         correlation_scores,
#         # voting,
#     )

# for i, j in enumerate(correlation_scores):
#     correlation_scores[i] = correlation_scores[i] / 5
# print(correlation_scores)


into = 0
for train_index, val_index in kf.split(x_train, y_train):
    into += 1
    model = ENGNet22(
        4,
        num_features,
    )
    model_name = "ENGNet2"
    model.summary()

    x_train_k, x_val = x_train[train_index], x_train[val_index]
    y_train_k, y_val = y_train[train_index], y_train[val_index]

    layer_to_save_weights = model.layers[2]
    weights_to_save = layer_to_save_weights.get_weights()

    output_folder_cv = ""

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
                monitor="accuracy",
                mode="max",
                patience=15,
                restore_best_weights=True,
            ),
            tfk.callbacks.ReduceLROnPlateau(
                monitor="accuracy", mode="max", patience=15, factor=0.5
            ),
            model_checkpoint,
        ],
    ).history

    model.load_weights(checkpoint_path)

    # Plotting, da aggiiungere la loss
    best_epoch = np.argmax(history["accuracy"])
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
        history["accuracy"],
        label="Training accuracy",
        alpha=0.8,
        color="#ff7f0e",
    )
    plt.plot(
        history["accuracy"],
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
    y_val_pred = model.predict(x_val)
    y_val_pred_nohot = np.argmax(y_val_pred, axis=1)
    conf_matrix_val = confusion_matrix(y_val, y_val_pred_nohot)
    # cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(prediction, axis=-1))

    # Compute the classification metrics
    # Calculate testing metrics
    y_test_pred = model.predict(x_test)
    print(y_test_pred)
    print(y_test_pred)
    print(y_test_pred)
    y_test_pred_nohot = np.argmax(y_test_pred, axis=1)

    # Calculate testing metrics
    test_results["test_accuracy"].append(accuracy_score(y_test, y_test_pred_nohot))
    test_results["test_f1_score"].append(
        f1_score(y_test, y_test_pred_nohot, average="macro")
    )
    test_results["test_weighted_f1_score"].append(
        f1_score(y_test, y_test_pred_nohot, average="weighted")
    )
    test_results["test_confusion_matrix"].append(
        confusion_matrix(y_test, y_test_pred_nohot)
    )

    cv_results["val_accuracy"].append(accuracy_score(y_fold_val, y_val_pred_nohot))
    cv_results["val_f1_score"].append(f1)
    cv_results["val_weighted_f1_score"].append(weighted_f1)
    cv_results["val_confusion_matrix"].append(conf_matrix_val)
    cv_results["val_best_weights"].append(model.get_weights())

    cv_results["val_loss"].append(history.history["loss"])
    cv_results["train_acc_history"].append(history.history["accuracy"])
    cv_results["val_acc_history"].append(history.history["val_accuracy"])

    print("Test Accuracy:", test_results["test_accuracy"])
    print("Test F1 score:", test_results["test_f1_score"])
    print("Test weighted F1 score:", test_results["test_weighted_f1_score"])

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    x_axis_labels = ["prop-", "prop+", "touch", "noci "]
    y_axis_labels = ["prop-", "prop+", "touch", "noci "]
    sns.heatmap(
        conf_matrix_val.T,
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

print(predictions1)

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
print(np.argmax(final_prediction, axis=1))

cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(final_prediction, axis=1))

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
