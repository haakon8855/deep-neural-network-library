"""Haakoas"""


class Config():
    """
    Fetches the config from a file and returns it in json/py-dict format.
    """
    @staticmethod
    def get_config(config_file: str):
        """
        Fetches the config from the given configuration file path and returns
        a json-object/python-dictionary containing the values.
        """
        return {
            "loss":
            "cross_entropy",  # (global) loss function
            "lrate":
            0.1,  # (global) learning rate
            "wreg":
            0.001,  # (global) weight regularization rate
            "wrt":
            "L2",  # (global) weight regularization type
            "input":
            20,  # size of input layer (number of input nodes)
            "layers": [
                {
                    "size": 100,
                    "act": "relu",
                    "wr": (-0.1, 0.1),  # weight range
                },
                {
                    "size": 5,
                    "act": "relu",
                    "wr": (-0.5, 0.5),  # weight range
                },
            ],
        }
