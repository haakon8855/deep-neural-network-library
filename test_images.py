"""haakoas"""

from matplotlib import pyplot as plt

from network import Network
from configuration import Config
from data_generator import DataGenerator


class TestImages():
    """
    Test class for running network on generated images.
    """
    def __init__(self, config_file: str):
        self.config = Config.get_config(config_file)

        generator = DataGenerator(20, 5, 20, 5, 20, 0.02)
        self.train, self.validation, self.test = generator.generate_images(800)

        self.network = Network(self.config, self.train, self.validation,
                               self.test)

    def main(self):
        """
        Main method for running the images on the neural network.
        """
        # Print network layer structure
        print(self.network)

        # Run the validation set before training
        self.network.forward_pass(self.network.validation_x,
                                  self.network.validation_y,
                                  verbose=True,
                                  validation=True)

        # Train the network, defining epochs and batch size
        train_time = self.network.fit(epochs=100, batch_size=20)

        # Run the validation set after training
        self.network.forward_pass(self.network.validation_x,
                                  self.network.validation_y,
                                  verbose=True,
                                  validation=True)

        print(f"Time to train: {round(train_time, 3)}")  # Time used to train

        # Plot the loss curves for the training, validation and test data sets.
        plt.xlabel("Minibatch")
        plt.ylabel("Loss")
        plt.plot(self.network.train_loss_index, self.network.train_loss)
        plt.plot(self.network.validation_loss_index,
                 self.network.validation_loss)
        plt.legend(["Train", "Validation"])
        plt.show()


if __name__ == "__main__":
    test_images = TestImages("config.ini")
    test_images.main()
