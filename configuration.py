"""Haakoas"""


class Config():
    """Fetches the config from a file and returns it in json/py-dict format."""
    @staticmethod
    def get_config(config_file: str):
        return {
            "loss": "cross_entropy",
            "lrate": 0.1,
            "wreg": 0.001,
            "wrt": "L2",
            "input": 20,
            "layer1": {
                "size": 100,
                "act": "relu",
                "wr": (-0.1, 0.1),  # weight range
                "lrate": 0.01,
            },
            "layer2": {
                "size": 5,
                "act": "relu",
                "wr": "glorot",  # weight range
                "br": (0, 1),  # bias range
            },
            "layer3": {
                "type": "softmax",
            },
        }