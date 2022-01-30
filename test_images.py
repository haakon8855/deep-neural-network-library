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
        self.epochs = int(self.config['GLOBALS']['epochs'])
        self.batch_size = int(self.config['GLOBALS']['batch_size'])
        self.verbose = self.config['GLOBALS']['verbose'] == 'true'
        self.centered = self.config['GLOBALS']['centered'] == 'true'

        generator = DataGenerator(20,
                                  10,
                                  20,
                                  10,
                                  20,
                                  0.01,
                                  centered=self.centered)
        self.train, self.validation, self.test = generator.generate_images(800)

        self.network = Network(self.config, self.train, self.validation,
                               self.test)
        if self.config['GLOBALS']['show_images'] == 'true':
            generator.visualize_all()

    def main(self):
        """
        Main method for running the images on the neural network.
        """
        # Print network layer structure
        print(self.network)

        # Run the validation set before training
        self.network.forward_pass(self.network.validation_x,
                                  self.network.validation_y,
                                  verbose=self.verbose,
                                  data_set=1)

        # Train the network, defining epochs and batch size
        train_time = self.network.fit(epochs=self.epochs,
                                      batch_size=self.batch_size)

        # Run the validation set after training
        self.network.forward_pass(self.network.validation_x,
                                  self.network.validation_y,
                                  verbose=self.verbose,
                                  data_set=1)

        print(f"Time to train: {round(train_time, 3)}")  # Time used to train

        # Run the test set after training
        self.network.forward_pass(self.network.test_x,
                                  self.network.test_y,
                                  verbose=False,
                                  data_set=2)

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
