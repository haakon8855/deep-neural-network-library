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

        # Fetch network config
        global_conf = self.config['GLOBALS']
        self.epochs = int(global_conf['epochs'])
        self.batch_size = int(global_conf['batch_size'])
        self.verbose = global_conf['verbose'] == 'true'
        self.wrt = global_conf['wrt']
        self.wreg = float(global_conf['wreg'])

        # Fetch image generation config
        self.image_size = int(global_conf['image_size'])
        self.min_width = int(global_conf['min_width'])
        self.max_width = int(global_conf['max_width'])
        self.min_height = int(global_conf['min_height'])
        self.max_height = int(global_conf['max_height'])
        self.noise = float(global_conf['noise'])
        self.centered = global_conf['centered'] == 'true'
        self.data_set_size = int(global_conf['data_set_size'])
        self.train_size = float(global_conf['train_size'])
        self.valid_size = float(global_conf['valid_size'])

        # Initialize data generator
        generator = DataGenerator(self.image_size,
                                  self.min_width,
                                  self.max_width,
                                  self.min_height,
                                  self.max_height,
                                  self.noise,
                                  split=(self.train_size, self.valid_size, 1 -
                                         (self.train_size + self.valid_size)),
                                  centered=self.centered)
        self.train, self.validation, self.test = generator.generate_images(
            self.data_set_size)

        # Initialize network
        self.network = Network(self.config,
                               self.train,
                               self.validation,
                               self.test,
                               wrt=self.wrt,
                               wreg=self.wreg)
        if global_conf['show_images'] == 'true':
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
    test_images = TestImages("config1.ini")  # 3 hidden layers
    # test_images = TestImages("config2.ini")  # 0 hidden layers
    # test_images = TestImages("config3.ini")  # 5 hidden layers

    # # Identical to config1.ini but without softmax, and mse instead of cross_entropy
    # test_images = TestImages("config4.ini")  # 3 hidden layers
    test_images.main()
