import numpy as np
import math
import os
import scipy.io
import glob
from matplotlib import pyplot as plt

from toolbox.functions.Functions import load_npz, extract_window_size, calc_mean_std


def plot_history(path, history, fold):
    # Loss progress during training
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure('Loss')
    plt.plot(epochs, loss, 'c', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/'+'Loss_CV'+str(fold+1)+'.jpg')
    plt.close()
    plt.show()

    # Accuracy progress during training
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    plt.figure('Accuracy')
    plt.plot(epochs, acc, 'c', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path+'/'+'Accuracy_CV'+str(fold+1)+'.jpg')
    plt.close()
    #plt.show()

    
    
def plot_results_2(path, y_true,y_pred,fold):
    # Accuracy and F1-Score for the test set
    f1_score=metrics.f1_score(y_true, y_pred, average='macro')
    f1_score_w=metrics.f1_score(y_true, y_pred, average='weighted')
    accuracy=metrics.accuracy_score(y_true, y_pred)
    print ("Accuracy = ", accuracy)
    print ("F1_score = ", f1_score)
    print ("F1_score weighted= ", f1_score_w)
    
    f = open(path+'accuracy_f1score_CV'+str(fold+1)+'.txt', 'w')
    f.write('Accuracy = '+str(accuracy)+'\n')
    f.write('F1_score = '+str(f1_score)+'\n')
    f.write('F1_score weighted = '+str(f1_score_w)+'\n')
    f.close()
    
    # Confusion matrix for the test set
    cm = confusion_matrix(y_true, y_pred)
    plt.figure('Confusion Matrix')
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues")
    plt.savefig(path+'Confusion_Matrix_CV'+str(fold+1)+'.jpg')
    plt.close()## Evaluation functions    
    
    
    
def plot_results(path, y_true,y_pred,fold,train_text,time_taken):
    # Accuracy and F1-Score for the test set
    f1_score=metrics.f1_score(y_true, y_pred, average='macro')
    f1_score_w=metrics.f1_score(y_true, y_pred, average='weighted')
    accuracy=metrics.accuracy_score(y_true, y_pred)
    print ("Accuracy = ", accuracy)
    print ("F1_score = ", f1_score)
    print ("F1_score weighted= ", f1_score_w)
    
    if (time_taken!=0):
        f = open(path+'accuracy_f1score_CV'+str(fold+1)+'.txt', 'w')
    else:
        f = open(path+'accuracy_f1score_CV'+str(fold+1)+'.txt', 'a')
        
    f.write(train_text+'Accuracy = '+str(accuracy)+'\n')
    f.write(train_text+'F1_score = '+str(f1_score)+'\n')
    f.write(train_text+'F1_score weighted = '+str(f1_score_w)+'\n')
    
    if (time_taken!=0):
        f.write('Time taken = '+str(time_taken)+'s\n')
        f.write('Time taken/sample = '+str(1000*time_taken/len(y_pred))+'ms\n')
    f.close()
    
    # Confusion matrix for the test set
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(train_text+'Confusion Matrix')
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues")
    plt.savefig(path+train_text+'Confusion_Matrix_CV'+str(fold+1)+'.jpg')
    plt.close()
    
# Function to create a boxplot for accuracy
def create_accuracy_boxplot(accuracy_data, window_sizes, animale, data_directory,  T_or_V, ymin ):
    # Create the boxplot for accuracy
    # Definiamo la larghezza dei boxplot
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.boxplot(accuracy_data, widths=width)
    plt.xticks(range(1, len(window_sizes) + 1), window_sizes)
    plt.xlabel('Window Size (ms)')
    plt.ylabel('Accuracy (%)')
    title = f'{T_or_V} Accuracy Comparison for {animale}'
    plt.title(title, fontsize=14)
    plt.ylim(ymin, 100)
    plt.grid(True)

    # Annotate mean and standard deviation on the boxplots
    for i, acc_data in enumerate(accuracy_data):
        mean_accuracy = np.mean(acc_data)
        std_accuracy = np.std(acc_data, ddof=1)
        plt.text(i + 0.75, mean_accuracy -2, f'{mean_accuracy:.1f}\n± {std_accuracy:.1f}', fontsize=12, ha='center', va='center')

    plt.savefig(f"{data_directory}/{title}.png")
    plt.show()

# Function to create a boxplot for F1-score

def create_f1score_boxplot(f1score_data, window_sizes, animale, data_directory,  T_or_V, ymin):
    # Create the boxplot for F1-score
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.boxplot(f1score_data, widths=width)
    plt.xticks(range(1, len(window_sizes) + 1), window_sizes)
    plt.xlabel('Window Size (ms)')
    plt.ylabel('F1-score (%)')
    title = f'{T_or_V} F1-score Comparison for {animale}'
    plt.title(title, fontsize=14)
    plt.ylim(ymin, 100)
    plt.grid(True)

    # Annotate mean and standard deviation on the boxplots
    for i, f1s_data in enumerate(f1score_data):
        mean_f1score = np.mean(f1s_data)
        std_f1score = np.std(f1s_data, ddof=1)
        plt.text(i + 0.75, mean_f1score - 2, f'{mean_f1score:.1f}\n± {std_f1score:.1f}', fontsize=12, ha='center', va='center')

    plt.savefig(f"{data_directory}/{title}.png")
    plt.show()

def get_animal_data(path_classifier, T_or_V):
    
    # Dictionary to store the data for each animal
    animal_data = {}
    
    # Loop through the animal folders and load the data
    for animale_folder in ['Animal 1', 'Animal 2', 'Animal 3']:
        
        data_directory=path_classifier+'/'+ animale_folder

        # Get a list of all .npz files in the directory
        file_list = os.listdir(data_directory)
        npz_files = [file_name for file_name in file_list if file_name.endswith('.npz')]
        
        # Sort npz_files in descending order based on the window size
        npz_files.sort(key=extract_window_size, reverse=True)
        
        npz_files = [file for file in npz_files if T_or_V in file]
        # Lists to store window sizes, accuracy, and f1score data
        window_sizes = []
        accuracy_data = []
        f1score_data = []

        # Load data from each .npz file and collect the window sizes, accuracy, and f1score
        for file_name in npz_files:
            accuracy, f1score = load_npz(os.path.join(data_directory, file_name))
            window_sizes.append(extract_window_size(file_name))
            accuracy_data.append(accuracy * 100)
            f1score_data.append(f1score * 100)

        # Store the data in the animal_data dictionary
        animal_data[animale_folder] = {
            'window_sizes': window_sizes,
            'accuracy_data': accuracy_data,
            'f1score_data': f1score_data
        }

    return animal_data


def w_compare_acc_boxplots_same_wind(accuracy_w100_1, accuracy_w100_2, accuracy_w100_3, path_boxplots, T_or_V, ymin):
    
    #definiamo le posizioni dei boxplot
    positions= [1, 2, 3, 5, 6, 7, 9, 10, 11]
    
    #creazione delle etichette che si ripetono:
    
    tags = ['Animal 1', 'Animal 2', 'Animal 3']
    
    # Def boxplots lenght
    width = 0.4
    
    plt.figure(figsize=(20,10))
    
    # Colori dei boxplot
    box_colors = ['lightblue', 'lightgreen', 'lightyellow']
    
    # Boxplot per animal_1_100ms
    boxplot_100ms_1 = plt.boxplot(accuracy_w100_1, positions=positions[6:9], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[2]))
    
    # Boxplot per animal_2_100ms
    boxplot_100ms_2 = plt.boxplot(accuracy_w100_2, positions=positions[6:9], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[2]))
    
    # Boxplot per animal_3_100ms
    boxplot_100ms_3 = plt.boxplot(accuracy_w100_3, positions=positions[6:9], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[2]))
    

    # Assegniamo i colori ai boxplot in base ai gruppi di dati
    for boxplot, color in zip([boxplot_100ms_1, boxplot_100ms_2, boxplot_100ms_3], box_colors):
        for box in boxplot['boxes']:
            box.set(facecolor=color)

    plt.ylabel('Accuracy (%)', fontsize=16)
    title=f'{T_or_V} Boxplot of Accuracies, same window comparison'
    plt.title(title, fontsize=16)
    plt.ylim(ymin, 100)
    plt.yticks(fontsize=14)
    plt.xticks([])
    plt.grid(True)
    
 # Annotazione media e deviazione standard per ciascun boxplot
    all_accuracy_data = accuracy_w100_1 + accuracy_w100_2 + accuracy_w100_3
    for i, accuracy_data in enumerate(all_accuracy_data):
        mean = np.mean(accuracy_data)
        std = np.std(accuracy_data, ddof=1)
        plt.text(positions[i]-0.4, mean, f'{mean:.1f}\n±{std:.1f}', ha='center', va='center', fontsize=12)
        plt.text(positions[i], ymin-2, etichette[i], ha='center', va='center', fontsize=12)  # Aggiungo l'etichetta

    plt.text(2, ymin-4, 'Animal 1 100ms', ha='center', va='center', fontsize=12)
    plt.text(6, ymin-4, 'Animal 2 100ms', ha='center', va='center', fontsize=12)
    plt.text(10, ymin-4, 'Animal 3 100ms', ha='center', va='center', fontsize=12)
 
    plt.savefig(f"{path_boxplots}/{title}.png")
    plt.show()





def w_compare_acc_boxplots(accuracy_w500, accuracy_w200, accuracy_w100, accuracy_w50, path_boxplots, T_or_V, ymin):


    # Definiamo le posizioni dei boxplot
    positions = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]

    # Creazione delle etichette che si ripetono
    etichette = ['Animal 1',  'Animal 2', 'Animal 3', 'Animal 1',  'Animal 2', 'Animal 3', 'Animal 1',  'Animal 2', 'Animal 3', 'Animal 1',  'Animal 2', 'Animal 3' ]

    # Definiamo la larghezza dei boxplot
    width = 0.4

    plt.figure(figsize=(20, 10))

    # Colori dei boxplot
    box_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']

    # Boxplot per 500ms
    boxplot_500ms = plt.boxplot(accuracy_w500, positions=positions[:3], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[0]))

    # Boxplot per 200ms
    boxplot_200ms = plt.boxplot(accuracy_w200, positions=positions[3:6], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[1]))

    # Boxplot per 100ms
    boxplot_100ms = plt.boxplot(accuracy_w100, positions=positions[6:9], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[2]))

    # Boxplot per 50ms
    boxplot_50ms = plt.boxplot(accuracy_w50, positions=positions[9:], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[3]))

    # Assegniamo i colori ai boxplot in base ai gruppi di dati
    for boxplot, color in zip([boxplot_500ms, boxplot_200ms, boxplot_100ms, boxplot_50ms], box_colors):
        for box in boxplot['boxes']:
            box.set(facecolor=color)

    plt.ylabel('Accuracy (%)', fontsize=16)
    title=f'{T_or_V} Boxplot of Accuracies, windows comparison'
    plt.title(title, fontsize=16)
    plt.ylim(ymin, 100)
    plt.yticks(fontsize=14)
    plt.xticks([])
    plt.grid(True)

    # Annotazione media e deviazione standard per ciascun boxplot
    all_accuracy_data = accuracy_w500 + accuracy_w200 + accuracy_w100 + accuracy_w50
    for i, accuracy_data in enumerate(all_accuracy_data):
        mean = np.mean(accuracy_data)
        std = np.std(accuracy_data, ddof=1)
        plt.text(positions[i]-0.4, mean, f'{mean:.1f}\n±{std:.1f}', ha='center', va='center', fontsize=12)
        plt.text(positions[i], ymin-2, etichette[i], ha='center', va='center', fontsize=12)  # Aggiungo l'etichetta

    plt.text(2, ymin-4, '500ms', ha='center', va='center', fontsize=12)
    plt.text(6, ymin-4, '200ms', ha='center', va='center', fontsize=12)
    plt.text(10, ymin-4, '100ms', ha='center', va='center', fontsize=12)
    plt.text(14, ymin-4, '50ms', ha='center', va='center', fontsize=12)

    plt.savefig(f"{path_boxplots}/{title}.png")
    plt.show()



def w_compare_f1_boxplots(f1score_w500, f1score_w200, f1score_w100, f1score_w50, path_boxplots, T_or_V, ymin):


    # Definiamo le posizioni dei boxplot
    positions = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]

    # Creazione delle etichette che si ripetono
    etichette = ['Animal 1',  'Animal 2', 'Animal 3', 'Animal 1',  'Animal 2', 'Animal 3', 'Animal 1',  'Animal 2', 'Animal 3', 'Animal 1',  'Animal 2', 'Animal 3' ]

    # Definiamo la larghezza dei boxplot
    width = 0.4

    plt.figure(figsize=(20, 10))

    # Colori dei boxplot
    box_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']

    # Boxplot per 500ms
    boxplot_500ms = plt.boxplot(f1score_w500, positions=positions[:3], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[0]))

    # Boxplot per 200ms
    boxplot_200ms = plt.boxplot(f1score_w200, positions=positions[3:6], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[1]))

    # Boxplot per 100ms
    boxplot_100ms = plt.boxplot(f1score_w100, positions=positions[6:9], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[2]))

    # Boxplot per 50ms
    boxplot_50ms = plt.boxplot(f1score_w50, positions=positions[9:], widths=width, patch_artist=True, boxprops=dict(facecolor=box_colors[3]))

    # Assegniamo i colori ai boxplot in base ai gruppi di dati
    for boxplot, color in zip([boxplot_500ms, boxplot_200ms, boxplot_100ms, boxplot_50ms], box_colors):
        for box in boxplot['boxes']:
            box.set(facecolor=color)

    plt.ylabel('f1score (%)', fontsize=16)
    title=f'{T_or_V} Boxplot of F1-scores, windows comparison'
    plt.title(title, fontsize=16)
    plt.ylim(ymin, 100)
    plt.yticks(fontsize=14)
    plt.xticks([])
    plt.grid(True)

    # Annotazione media e deviazione standard per ciascun boxplot
    all_f1score_data = f1score_w500 + f1score_w200 + f1score_w100 + f1score_w50
    for i, f1score_data in enumerate(all_f1score_data):
        mean = np.mean(f1score_data)
        std = np.std(f1score_data, ddof=1)
        plt.text(positions[i]-0.4, mean, f'{mean:.1f}\n±{std:.1f}', ha='center', va='center', fontsize=12)
        plt.text(positions[i], ymin-2, etichette[i], ha='center', va='center', fontsize=12)  # Aggiungo l'etichetta

    plt.text(2, ymin-4, '500ms', ha='center', va='center', fontsize=12)
    plt.text(6, ymin-4, '200ms', ha='center', va='center', fontsize=12)
    plt.text(10, ymin-4, '100ms', ha='center', va='center', fontsize=12)
    plt.text(14, ymin-4, '50ms', ha='center', va='center', fontsize=12)

    plt.savefig(f"{path_boxplots}/{title}.png")
    plt.show()


















