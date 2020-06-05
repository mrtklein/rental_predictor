from data_script.data import Data


class Trainer:
    def __init__(self):
        self.data = Data()

    def train(self):
        x_train, x_test, y_train, y_test = self.data.get_data()
