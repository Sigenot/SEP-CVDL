import pandas as pd

def create_csv(eval_true_labels_per_epoch, eval_pred_labels_per_epoch): #, train_losses, test_losses, eval_losses, train_accuracies, test_accuracies, eval_accuracies):
    # use length of lists to create epoch 
    #epochs = len(train_losses)
    """
    df_metrics = pd.DataFrame({
        'Epoch': range(1,epochs + 1),
        'Train Loss': train_losses,
        'Test Loss': test_losses,
        'Evaluation Loss': eval_losses,
        'Train Accuracy': train_accuracies,
        'Test Loss': test_accuracies,
        'Evaluation Accuracy': eval_accuracies
    })
    """
    # convert lists of labels to strings for each epoch
    true_labels_str = [', '.join(map(str, labels)) for labels in eval_true_labels_per_epoch]
    pred_labels_str = [', '.join(map(str, labels)) for labels in eval_pred_labels_per_epoch]


    def_labels = pd.DataFrame({
        #'Epoch': range(1, epochs + 1),
        'Evaluation True Labels': true_labels_str,
        'Evaluation Predicted Labels': pred_labels_str
    })

    # merge dataframes
    #df = pd.concat([df_metrics, def_labels['Evaluation True Labels'], def_labels['Evaluation Predicted Labels']], axis=1)

    def_labels.to_csv('prel_results_model.csv', index=False)




