"""
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os

# get the csv file
file = pd.read_csv('results_model.csv')
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

# check if csv file is available:
if not os.path.exists('results_model.csv'):
    print(f"Warning, could not read {file}")

# function to get label from datapath
def get_label(filepath):
    return filepath.split('_')[2].split('.')[0]

file['true'] = file['filepath'].apply(get_label)

# predicted label as class with highest score
file['predicted'] = file[emotions].idxmax(axis=1)

confusion_matrix = pd.crosstab(file['true'], file['predicted'], rownames=['True'], colnames=['Predicted'])

# ensure order of classes
confusion_matrix = confusion_matrix.reindex(index=emotions, columns=emotions, fill_value=0)

# plot heatmap
plt.figure(figsize=(9,7))
ax = sb.heatmap(confusion_matrix, fmt="d", cmap="Blues", square=True, cbar=True, annot=True)

# save plt
plt.savefig('heatmap.png', dpi=400, bbox_inches = 'tight', pad_inches=0.1)

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    # Erstellen der Konfusionsmatrix
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))

    # Plot der Konfusionsmatrix
    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                     xticklabels=class_names, yticklabels=class_names)

    # Anpassen der Textfarbe für bessere Lesbarkeit
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            text_color = 'white' if val > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, val, color=text_color, ha='center', va='center')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Beispielaufruf der Funktion
# plot_confusion_matrix(eval_true_labels, eval_pred_labels, class_names)
