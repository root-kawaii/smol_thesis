import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight


# from mne.decoding import UnsupervisedSpatialFilter
# from sklearn.decomposition import PCA, FastICA
import tensorflow as tf
import seaborn as sns
from modelz import *

# from intervals_mat import *

from datetime import datetime

from builtins import range
from utils import *


tfk = tf.keras
tfkl = tf.keras.layers

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
channel_bool = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]


num_features = 0
for i in channel_bool:
    if i:
        num_features += 1

# ..
num_electrodes = 0
check = 0
f = open("results.txt", "a")
current_time = datetime.now()

path_folder = "../data/100ms_2/"
file_name = [f for f in os.listdir("../data/100ms_2/")]
file_paths = []
for file_number in range(len(file_name)):
    file = os.path.join(path_folder, file_name[file_number])
    file_paths.append(file)


print(len(file_name))

x_samp, y_samp = transform_data_not_transpose(file_name, file_paths, 16)


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

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_samp), y=y_samp
)
# Convert class weights to a dictionary
class_weights_dict = dict(enumerate(class_weights))


# new = scipy.signal.correlate(x_samp_tt[11][2], x_samp_tt[10][2], mode="full")
# gh = argmax(new)
# plt.plot(new)
# plt.axvline(x=gh, color="r", linestyle="--")
# plt.show()

# labels_correlation_windowless = print_value_counts(y_samp)
# lengths = []
# for kk in labels_correlation_windowless.values():
#     lengths.append(kk)
# print(lengths)
# # Split the data_list based on split_indices
# data_merge = split_list_by_lengths(x_samp_tt, lengths)

# new_reduced_labels = []
# for i in range(len(data_merge)):
#     splice_index = int(len(data_merge[i]) * 0.50)
#     data_merge[i] = data_merge[i][0:splice_index]
#     new_reduced_labels.extend([i] * splice_index)

# print(len(data_merge[2]))
# x_samp_tt = np.concatenate(data_merge)
# print(x_samp_tt.shape)
# new_reduced_labels = np.array(new_reduced_labels)

x_train, x_test, y_train, y_test = train_test_split(
    x_samp_tt, y_samp, train_size=train_ratio, shuffle=True, random_state=42
)

# y_samp_2 = []
# print(y_samp.shape)
# for i, item in enumerate(y_samp):
#     y_samp_2.extend([item] * 500)
# y_samp_2 = np.array(y_samp_2)

# print(y_samp_2.shape)

# x_samp_4 = x_samp_4.transpose()


# voting = {}
# for j in range(17):
#     voting[j] = 0
# print(voting)

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
#         labels_correlation_windowless,
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
    model = EEGNet100(
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
        monitor="val_accuracy",
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
        epochs=120,
        validation_data=(x_val, y_val),
        # class_weight=class_weights_dict,
        callbacks=[
            tfk.callbacks.EarlyStopping(
                monitor="accuracy",
                mode="max",
                patience=20,
                restore_best_weights=True,
            ),
            tfk.callbacks.ReduceLROnPlateau(
                monitor="accuracy", mode="max", patience=20, factor=0.5
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
    f1 = f1_score(y_val, y_val_pred_nohot, average="macro")
    weighted_f1 = f1_score(y_val, y_val_pred_nohot, average="weighted")
    print("Test Accuracy:", test_results["test_accuracy"])
    print("Test F1 score:", test_results["test_f1_score"])
    print("Test weighted F1 score:", test_results["test_weighted_f1_score"])
    cv_results["val_accuracy"].append(accuracy_score(y_val, y_val_pred_nohot))
    cv_results["val_f1_score"].append(f1)
    cv_results["val_weighted_f1_score"].append(weighted_f1)
    cv_results["val_confusion_matrix"].append(conf_matrix_val)
    cv_results["val_best_weights"].append(model.get_weights())

    # cv_results["val_loss"].append(history.history["loss"])
    # cv_results["train_acc_history"].append(history.history["accuracy"])
    # cv_results["val_acc_history"].append(history.history["val_accuracy"])

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
