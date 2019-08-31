import matplotlib.pyplot as plt


def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# load the result

# show_train_history(history, 'acc', 'val_acc')
# show_train_history(history, 'loss', 'val_loss')
