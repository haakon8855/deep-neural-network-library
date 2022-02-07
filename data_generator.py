"""haakoas"""

import random
from bresenham import bresenham
import numpy as np
from matplotlib import pyplot as plt


class DataGenerator():
    """
    Generator class for generating data sets of nxn images.
    """
    def __init__(self,
                 n,
                 min_width,
                 max_width,
                 min_height,
                 max_height,
                 noise,
                 split=(0.7, 0.2, 0.1),
                 centered=True,
                 seed=None):
        if (n < 10 or n > 50):
            print(f"Parameter not within range: n={n}!")
        if seed is not None:
            random.seed(seed)  # Set random seed if specified
        self.size = n
        self.min_dim = (min_width, min_height)
        self.max_dim = (max_width, max_height)
        self.noise = noise
        self.split = split
        self.centered = centered
        self.images = None
        self.images_flattened = None
        self.classes = None

    def get_images(self, flattened=True):
        """
        Returns the generated images. Returns None if self.generate_images()
        has not been run yet.
        """
        if self.images is None:
            return None
        if flattened:
            source_list = self.images_flattened
        else:
            source_list = self.images
        # Split image list into train, validation and test sets before returning
        after_train = len(source_list) * self.split[0]
        after_validation = len(source_list) * (self.split[0] + self.split[1])
        train_x = np.array(source_list[:int(after_train)])
        train_y = np.array(self.classes[:int(after_train)])
        validation_x = np.array(
            source_list[int(after_train):int(after_validation)])
        validation_y = np.array(
            self.classes[int(after_train):int(after_validation)])
        test_x = np.array(source_list[int(after_validation):])
        test_y = np.array(self.classes[int(after_validation):])
        return (train_x, train_y), (validation_x, validation_y), (test_x,
                                                                  test_y)

    def generate_images(self, amount, flattened=True):
        """
        Generates a set of images according to the specified parameters.
        Then returns the images in the form of three data sets with
        corresponding truth values.
        """
        self.images = []
        self.images_flattened = []
        self.classes = []
        for _ in range(amount):
            image, classes = self.generate_one_image()
            self.images.append(image)
            self.images_flattened.append(image.flatten())
            # Use one-hot encoding for image classes
            classification = [0, 0, 0, 0]
            classification[classes] = 1
            self.classes.append(classification)
        return self.get_images(flattened)

    def generate_one_image(self):
        """
        Generates one single image and returns it as a 2d numpy array. The class
        (and thus the figures present in the image) is determined randomly.
        """
        image_class = random.randint(0, 3)
        if image_class == 0:
            image = self.generate_circle()
        elif image_class == 1:
            image = self.generate_rectangle()
        elif image_class == 2:
            image = self.generate_triangle()
        else:
            image = self.generate_cross()
        image = self.add_noise(image)
        return image, image_class

    def get_bounding_box(self):
        """
        Chooses a bounding box surrounding the figure. This determines the
        figure's size as the box's size is set randomly. Its position is also
        chosen randomly such that the entire box is within the image.
        """
        width = random.randint(self.min_dim[0], self.max_dim[0])
        height = random.randint(self.min_dim[1], self.max_dim[1])
        if self.centered:
            box_x = (self.size - width) // 2
            box_y = (self.size - height) // 2
        else:
            box_x = random.randint(0, self.size - width)
            box_y = random.randint(0, self.size - height)
        return box_x, box_y, width, height

    def generate_circle(self):
        """
        Generates one single image containing a circle and returns it as a
        2d numpy array.
        """
        box_x, box_y, box_width, box_height = self.get_bounding_box()
        radius = min(box_width, box_height) // 2
        center = np.array(
            [box_x + (box_width // 2), box_y + (box_height // 2)])

        image = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                # Circle equation to color the correct pixels
                circle = np.abs((i - center[0])**2 + (j - center[1])**2 -
                                radius**2)
                if circle <= radius:
                    image[i][j] = 1
        return image

    def generate_rectangle(self):
        """
        Generates one single image containing a rectangle and returns it as a
        2d numpy array.
        """
        box_x, box_y, box_width, box_height = self.get_bounding_box()

        image = np.zeros((self.size, self.size))
        # Horizontal lines in the rectangle
        image[box_y][box_x:box_x + box_width - 1] = 1
        image[box_y + box_height - 1][box_x:box_x + box_width] = 1
        # Vertical lines in the rectangle
        for y_pos in range(box_y, box_y + box_height):
            image[y_pos][box_x] = 1
            image[y_pos][box_x + box_width - 1] = 1
        return image

    def generate_triangle(self):
        """
        Generates one single image containing a triangle and returns it as a
        2d numpy array.
        """
        box_x, box_y, box_width, box_height = self.get_bounding_box()
        point_a = (box_y, (box_x + box_x + box_width) // 2
                   )  # Upper middle of rect
        point_b = (box_y + box_height - 1, box_x + box_width - 1
                   )  # Lower right of rect
        point_c = (box_y + box_height - 1, box_x)  # Lower left of rect

        image = np.zeros((self.size, self.size))
        # Use external library bresenham to color the correct pixels
        a_to_b = list(bresenham(point_a[1], point_a[0], point_b[1],
                                point_b[0]))
        b_to_c = list(bresenham(point_b[1], point_b[0], point_c[1],
                                point_c[0]))
        c_to_a = list(bresenham(point_c[1], point_c[0], point_a[1],
                                point_a[0]))
        for point in a_to_b + b_to_c + c_to_a:
            image[point[1]][point[0]] = 1
        return image

    def generate_cross(self):
        """
        Generates one single image containing a cross and returns it as a
        2d numpy array.
        """
        box_x, box_y, box_width, box_height = self.get_bounding_box()
        vertical_x = (box_x + (box_x + box_width)) // 2
        horizontal_y = (box_y + (box_y + box_height)) // 2

        image = np.zeros((self.size, self.size))
        # Horizontal line
        image[horizontal_y][box_x:box_x + box_width] = 1
        # Vertical line
        for y_pos in range(box_y, box_y + box_height):
            image[y_pos][vertical_x] = 1
        return image

    def add_noise(self, image):
        """
        Adds noise to an image according to the noise level parameter.
        """
        # Number of pixels to flip
        noise_pixels = int(self.noise * (self.size**2))
        for _ in range(noise_pixels):
            rand_x = random.randint(0, self.size - 1)
            rand_y = random.randint(0, self.size - 1)
            image[rand_y][rand_x] = 1 - image[rand_y][rand_x]
        return image

    def visualize_all(self):
        """
        Visualizes the images in each data set by plotting them using
        matplotlib.
        """
        train, validation, test = self.get_images(flattened=False)
        # data_set[0] is image, data_set[1] is class
        self.visualize_one_data_set(train[0], 'Training set')
        self.visualize_one_data_set(validation[0], 'Validation set')
        self.visualize_one_data_set(test[0], 'Test set')

    def visualize_one_data_set(self, data_set, title: str):
        """
        Visualizes the images in one specified data set using matplotlib.
        """
        fig, axs = plt.subplots(2, 5)
        fig.canvas.set_window_title(title)
        fig.suptitle(title)
        for i, axis in enumerate(axs.flat):
            axis.imshow(data_set[i])
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    GEN = DataGenerator(20, 5, 20, 5, 20, 0.01)
    (TRAIN_X,
     TRAIN_Y), (VAL_X, VAL_Y), (TEST_X,
                                TEST_Y) = GEN.generate_images(10,
                                                              flattened=False)
    for I, IMAGE in enumerate(TRAIN_X):
        print(TRAIN_Y[I])
        plt.imshow(IMAGE)
        plt.show()
