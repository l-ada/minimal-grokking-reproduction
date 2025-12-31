from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt


def plot_results(history):
    clear_output(wait=True) # Clear previous output for dynamic updating
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot Loss (Log Scale is better for Grokking)
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss (Log Scale)')
    ax1.legend()
    ax1.set_title('Grokking Progress')

    # Plot Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()

    # plt.savefig('grokking_plot.png') # Removed as per user request
    # print("Plot saved as grokking_plot.png") # Removed as per user request
    plt.show() # Display the updated plot
    plt.close(fig) # Close the figure to free up memory
