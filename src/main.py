from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from statistics import mean

from utils.cli_utils import cli_select, cli_text
from utils.validation_utils import is_float, is_integer
from utils.data_utils import Dataset
from perceptron import Perceptron


def main():
    print("Hi there, welcome to perceptron training program!")
    print("Let's do some prep work first!\n")
    config_dataset = cli_select(
        message="Please select a dataset to be used:",
        choices=["Iris dataset", "Breast Cancer dataset"]
    )
    config_activation_function = cli_select(
        message="Please select activation function:",
        choices=["Step Activatation", "Sigmoid Activation"]
    )
    config_learning_rate = cli_text(
        message="Please enter you desired learning rate:",
        validation_function=lambda x: is_float(x) and float(x) >= 0 and float(x) <= 1,
        transform_function=lambda x: float(x)
    )
    config_epoch_n = cli_text(
        message="Please enter you desired training epoch number:",
        validation_function=lambda x: is_integer(x) and float(x) > 0,
        transform_function=lambda x: int(x)
    )

    dataset = Dataset(config_dataset)
    perceptron = Perceptron(dataset.shape[1], config_learning_rate, config_activation_function)

    epoch_numbers = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for i in range(config_epoch_n):
        train_loss, train_acc, val_loss, val_acc = perceptron.fit(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val)

        print(f"Epoch {i + 1:<5} {'|':<5} train loss: {train_loss:<20} {'|':<5} val loss: {val_loss}")

        epoch_numbers.append(i + 1)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        

    # calculating final train / val accuracy
    y_train_pred = [perceptron.predict(x) for x in dataset.x_train]
    y_val_pred = [perceptron.predict(x) for x in dataset.x_val]
    final_train_acc = accuracy_score(dataset.y_train, y_train_pred)
    final_val_acc = accuracy_score(dataset.y_val, y_val_pred)

    # print(val_accuracies)
    # print(dataset.y_val)

    # printing the results
    print("\nTraining finished!")

    print("Final weights:\n",perceptron.w)

    print("\nStats for train datset")
    print(f"Accuracy: {final_train_acc:2}")
    print(f"Mean loss: {mean(train_losses):.2f}")

    print("\nStats for validation datset")
    print(f"Accuracy: {final_val_acc:2}")
    print(f"Mean loss: {mean(val_losses):.2f}")

    # plotting the loss/acc plots for training and validation
    plt.plot(epoch_numbers, train_losses, label = "train_loss")
    plt.plot(epoch_numbers, val_losses, label = "val_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 20)
    plt.show()

    plt.plot(epoch_numbers, train_accuracies, label = "train_accuracy")
    plt.plot(epoch_numbers, val_accuracies, label = "val_accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()