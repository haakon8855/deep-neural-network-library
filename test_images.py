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
        self.wrt = self.config['GLOBALS']['wrt']
        self.wreg = float(self.config['GLOBALS']['wreg'])

        self.image_size = int(self.config['GLOBALS']['image_size'])
        self.min_width = int(self.config['GLOBALS']['min_width'])
        self.max_width = int(self.config['GLOBALS']['max_width'])
        self.min_height = int(self.config['GLOBALS']['min_height'])
        self.max_height = int(self.config['GLOBALS']['max_height'])
        self.noise = float(self.config['GLOBALS']['noise'])
        self.centered = self.config['GLOBALS']['centered'] == 'true'
        self.data_set_size = int(self.config['GLOBALS']['data_set_size'])

        generator = DataGenerator(self.image_size,
                                  self.min_width,
                                  self.max_width,
                                  self.min_height,
                                  self.max_height,
                                  self.noise,
                                  centered=self.centered)
        self.train, self.validation, self.test = generator.generate_images(
            self.data_set_size)

        self.network = Network(self.config,
                               self.train,
                               self.validation,
                               self.test,
                               wrt=self.wrt,
                               wreg=self.wreg)
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
    test_images = TestImages("config1.ini")
    # test_images = TestImages("config2.ini")
    # test_images = TestImages("config3.ini")
    test_images.main()
