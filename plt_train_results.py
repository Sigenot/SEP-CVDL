import matplotlib.pyplot as plt


def plot_fun(train_losses, test_losses, eval_losses, train_accuracies, test_accuracies, eval_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.plot(epochs, eval_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses on model')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train accuracy') 
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.plot(epochs, eval_accuracies, label='Evaluation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracies on model')
    plt.legend()

    plt.savefig('data_plot.png')