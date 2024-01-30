import numpy as np
import math
import os
import scipy.io
import glob
from tensorflow.keras.utils import plot_model
import re

from toolbox.models.EEGModels import EEGNet, EEGNet2
from toolbox.models.CNNModels import CNN_temp_elec, CONV_LSTM, CNN_4x


# Imports sample and transforms it into a numpy array
def import_sample(path):
    mat = scipy.io.loadmat(path,mat_dtype=True)
    x = np.array(mat['to_save'])
    file_idx=int(mat['file_idx'])
    sample_idx=int(mat['sample_idx'])
    original_sample=int(mat['original_sample'])
    n_samples_original=int(mat['n_samples_original'])
    return x,file_idx,sample_idx,original_sample,n_samples_original

# Gets label from the name of the file
def get_label_from_path(name):
    
    # Remove last number if proprio - adaptation to Davide files
    # Define a regular expression pattern to match "_XX" where XX is a 2-digit number
    pattern = r"_[0-9]{2}$"
    if re.search(pattern, pattern):
        # Use re.sub to replace the matched pattern with an empty string
        result_string = re.sub(pattern, "", name)
       
    # Remove file extension
    label = name.replace(".mat", "")
    # Remove intial part of file name, eg "sample_0161_"
    label = label[12:]
    return label

# Encodes label depending on the name of the file
def encod_label(lab):
    if ('idle' in lab):
        label=4
    else:

        if (lab[0:4] == 'noci' or
            lab[0:4] == 'Noci' or
            lab[0:5] == 'Pinch'):
            label=0
        elif (lab[0:4] == 'prop' and
            lab[11] == '-'):
            label=1
        elif (lab[0:4] == 'prop'):
            label=2
        elif (lab[0:5] == 'touch'):
            label=3    
        
    return label

# Used to select each specific type of stimulus, diferentiating
# intensities of stimulation
def encod_label_tot(lab):
    if (lab[0:9] == 'noci_heel' or
        lab[0:16]=='noci_trial_heel'):
        label=0
    elif (lab[0:10] == 'noci_outer' or 
          lab[0:10]=='noci_pinky'):
        label=1
    elif (lab[0:16] == 'noci_trial_outer' or
          lab[0:16]=='noci_trial_pinky'):
        label=1
    elif (lab[0:4] == 'prop'):
        if (lab[11]=='-'):
            if (lab[12]=='1'):
                label=2
            elif (lab[12]=='2'):
                label=3
            else:
                label=4
        else:
            if (lab[11]=='1'):
                label=5
            elif (lab[11]=='2'):
                label=6
            else:
                label=7     
    elif (lab[0:5]=='touch'):
        if (lab[6]=='1'):
            label=8
        else:
            label=9
    return label

#Normalizes the data received
def normalise_data(max_value, data):
    x_norm=data/max_value
    return x_norm

# Imports all samples and saves them into x - sample - and y - target
def transform_data(file_name,file_paths,n_features):
    y_samp=np.empty(len(file_name))
    file_idx_vec=np.empty(len(file_name))
    sample_idx_vec=np.empty(len(file_name))
    original_sample_vec=np.empty(len(file_name))
    n_samples_original_vec=np.empty(len(file_name))
    sample=[]
    for file_number in range(len(file_name)):
        sample,file_idx,sample_idx,original_sample,n_samples_original=import_sample(file_paths[file_number])
        if (len(np.shape(sample))==3):
            # It's dealing with files that Elisa's old code generated, so we have to adequate it
            sample=sample[0,:,0:16]
        else:
            # Dealing with files that Elisa's new code, changed by Rafael, generated
            sample=np.transpose(sample) 
        if file_number==0:
            x_samp=np.empty((len(file_name),np.shape(sample)[0],n_features))
        # Gets the label from the file name
        lab=get_label_from_path(file_name[file_number])
        # Stores the sample
        x_samp[file_number,:,:]=sample
        # Encodes the label to a value that will be the target
        y_samp[file_number]=encod_label(lab)
        
        # Encodes if the sample was obtained via overlapping or not and the characteristics of it so
        # The training set can be built with no overlapping regarding the test set
        file_idx_vec[file_number]=file_idx
        sample_idx_vec[file_number]=sample_idx
        original_sample_vec[file_number]=original_sample
        n_samples_original_vec[file_number]=n_samples_original
        
        x_samp_trasposta = np.transpose(x_samp, (0, 2, 1))

        
    return x_samp_trasposta, y_samp, file_idx_vec, sample_idx_vec, original_sample_vec, n_samples_original_vec



# Imports all samples and saves them into x - sample - and y - target
def transform_data_transpose(file_name, file_paths, n_features):
    y_samp=np.empty(len(file_name))
    #file_idx_vec=np.empty(len(file_name))
    #sample_idx_vec=np.empty(len(file_name))
    #original_sample_vec=np.empty(len(file_name))
    #n_samples_original_vec=np.empty(len(file_name))
    sample=[]
    for file_number in range(len(file_name)):
            
        #file_idx,sample_idx,original_sample,n_sample_original
        
        print(file_number)
        sample = import_sample(file_paths[file_number])
        
        if (len(np.shape(sample))==3):
            # It's dealing with files that Elisa's old code generated, so we have to adequate it
            sample=sample[0,:,0:16]
        else:
            # Dealing with files that Elisa's new code, changed by Rafael, generated
            sample=np.transpose(sample) 
        if file_number==0:
            x_samp=np.empty((len(file_name),np.shape(sample)[0],n_features))
            
        print(file_name[file_number])    
        # Gets the label from the file name
        lab = get_label_from_path(file_name[file_number])
        print(lab)
        # Stores the sample
        x_samp[file_number,:,:]=sample
     
        # Encodes the label to a value that will be the target
        y_samp[file_number]=encod_label(lab)
        
        # Encodes if the sample was obtained via overlapping or not and the characteristics of it so
        # The training set can be built with no overlapping regarding the test set
        #file_idx_vec[file_number]=file_idx
        #sample_idx_vec[file_number]=sample_idx
        #original_sample_vec[file_number]=original_sample
        #n_samples_original_vec[file_number]=n_samples_original
        
        x_samp_trasposta = np.transpose(x_samp, (0, 2, 1))
           
        #, file_idx_vec, sample_idx_vec, original_sample_vec, n_samples_original_vec
            
    return x_samp_trasposta , y_samp


def transform_data_not_transpose(file_name, file_paths, n_features):
    y_samp=np.empty(len(file_name))
    #file_idx_vec=np.empty(len(file_name))
    #sample_idx_vec=np.empty(len(file_name))
    #original_sample_vec=np.empty(len(file_name))
    #n_samples_original_vec=np.empty(len(file_name))
    sample=[]
    for file_number in range(len(file_name)):
            
        #file_idx,sample_idx,original_sample,n_sample_original
        
        print(file_number)
        sample = import_sample(file_paths[file_number])
        
        if (len(np.shape(sample))==3):
            # It's dealing with files that Elisa's old code generated, so we have to adequate it
            sample=sample[0,:,0:16]
        else:
            # Dealing with files that Elisa's new code, changed by Rafael, generated
            sample=np.transpose(sample) 
        if file_number==0:
            x_samp=np.empty((len(file_name),np.shape(sample)[0],n_features))
            
        print(file_name[file_number])    
        # Gets the label from the file name
        lab = get_label_from_path(file_name[file_number])
        print(lab)
        # Stores the sample
        x_samp[file_number,:,:]=sample
     
        # Encodes the label to a value that will be the target
        y_samp[file_number]=encod_label(lab)
        
        # Encodes if the sample was obtained via overlapping or not and the characteristics of it so
        # The training set can be built with no overlapping regarding the test set
        #file_idx_vec[file_number]=file_idx
        #sample_idx_vec[file_number]=sample_idx
        #original_sample_vec[file_number]=original_sample
        #n_samples_original_vec[file_number]=n_samples_original
        
        #x_samp_trasposta = np.transpose(x_samp, (0, 2, 1))
           
        #, file_idx_vec, sample_idx_vec, original_sample_vec, n_samples_original_vec
            
    return x_samp, y_samp



def generate_data(num_samples, num_channels, num_time_points, num_classes):
    # Generate random signals for each class
    signals = []
    labels = []
    for i in range(num_classes):
        # Generate random signals for the current class
        class_signals = np.random.randn(num_samples, num_channels, num_time_points)
        signals.append(class_signals)
        
        # Assign labels to the generated signals
        class_labels = np.full((num_samples,), i)
        labels.append(class_labels)
    
    # Concatenate signals and labels for all classes
    signals = np.concatenate(signals, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return signals, labels


def create_model(path, net, n_classes, Samples, kernLength):
        
    name = path+ '/'+net+'model.png'
    
    if (net=='EEGNet'):
        
        model=EEGNet(nb_classes=n_classes,  Samples=Samples, kernLength=kernLength)
        plot_model(model, to_file=name, show_shapes=True,expand_nested=True, layer_range=None, show_layer_activations=True)
        
        model.summary()
       
           
        return model
    
    elif net == 'CNN_temp_elec':
        model = CNN_temp_elec(sequence_size=(Samples, 1), num_classes=n_classes)
        plot_model(model, to_file=name, show_shapes=True,expand_nested=True, layer_range=None, show_layer_activations=True)
        
        model.summary()
        
        return model
    
    elif net == 'CONV_LSTM':
        model = CONV_LSTM(sequence_size=(Samples,1), num_classes=n_classes)
        plot_model(model, to_file=name, show_shapes=True,expand_nested=True, layer_range=None, show_layer_activations=True)
        
        model.summary()
        
        return model
    
    elif (net=='CNN_4x'):
        
        model= CNN_4x(nb_classes=n_classes,  Samples=Samples, kernLength=kernLength)
        plot_model(model, to_file=name, show_shapes=True,expand_nested=True, layer_range=None, show_layer_activations=True)
        
        model.summary()
        
        return model
        
    else:
        raise ValueError('toolbox.Functions.create_models non conosce questa rete: '+ net)

        
def create_model_lstm(path, net, n_classes, sequence_size):

    name = path+ '/'+net+'model.png'

    if (net=='EEGNet'):

        model=EEGNet(nb_classes=n_classes,  Samples=Samples, kernLength=kernLength)
        plot_model(model, to_file=name, show_shapes=True,expand_nested=True, layer_range=None, show_layer_activations=True)

        model.summary()


        return model

    elif net == 'CNN_temp_elec':
        model = CNN_temp_elec(sequence_size=(Samples, 1), num_classes=n_classes)
        plot_model(model, to_file=name, show_shapes=True,expand_nested=True, layer_range=None, show_layer_activations=True)

        model.summary()

        return model

    elif net == 'CONV_LSTM':
        model = CONV_LSTM(sequence_size = sequence_size, num_classes=n_classes)
        plot_model(model, to_file=name, show_shapes=True,expand_nested=True, layer_range=None, show_layer_activations=True)

        model.summary()

        return model

    elif (net=='CNN_4x'):

        model= CNN_4x(nb_classes=n_classes,  Samples=Samples, kernLength=kernLength)
        plot_model(model, to_file=name, show_shapes=True,expand_nested=True, layer_range=None, show_layer_activations=True)

        model.summary()

        return model

    else:
        raise ValueError('toolbox.Functions.create_models non conosce questa rete: '+ net)

        
        
# Function to extract the window size from the filename
def extract_window_size(filename):
    return int(re.search(r'_(\d+)ms\.npz', filename).group(1))

def calc_mean_std(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return mean, std

    
# Function to load the data from .npz file
def load_npz(file_path):
    data = np.load(file_path)
    accuracy = data['accuracy']
    f1score = data['f1score']
    return accuracy, f1score

