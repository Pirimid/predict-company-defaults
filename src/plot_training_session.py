import matplotlib.pyplot as plt


def plot_session(train_loss_results, train_accuracy_results, save=False, name='image'):
    """
     Plots the training results to visualize the training accuracy and training loss.
      Arguments:
       * train_loss_results- Training loss results
       * train_accuracy_results- Training accuracy
    """
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()

    if save:
        fig.savefig(f'img/{name}.png')
        plt.close(fig)

def plot_test_accuracy(test_accuracy_results, save=False, name='test_acc'):
    """
     Plots the testing results to visualize the testing accuracy.
      Arguments:
       * test_accuracy_results- Testing accuracy results
    """
    fig, axes = plt.subplots(sharex=True, figsize=(12, 8))
    fig.suptitle('Testing Metrics')

    axes.set_ylabel("Accuracy", fontsize=14)
    axes.set_xlabel("Epoch", fontsize=14)
    axes.plot(test_accuracy_results)
    plt.show()

    if save:
        fig.savefig(f'img/{name}.png')
        plt.close(fig)