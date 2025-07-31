import pandas as pd

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

# get saved csv file
file_path = 'prel_results_model.csv'
data = pd.read_csv(file_path)

# change numbers to word
def convert_labels_to_words(label_str):
    # convert string to list of numbers
    labels = list(map(int, label_str.split(', ')))
    return [emotions[label] for label in labels]

# exec function
data['Evaluation True Labels Words'] = data['Evaluation True Labels'].apply(convert_labels_to_words)
data['Evaluation Predicted Labels Words'] = data['Evaluation Predicted Labels'].apply(convert_labels_to_words)

# remove redundant data
data = data.drop(columns=['Evaluation True Labels', 'Evaluation Predicted Labels'])

# save in csv file
data.to_csv('results_model.csv', index=False)