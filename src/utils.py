import numpy as np
import scipy


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


# x = 1
# print(x.shape())
