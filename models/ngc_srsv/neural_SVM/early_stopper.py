import numpy as np


class EarlyStopper:
    PATIENCE_ENDED = 1
    PATIENCE_STARTED = 2
    EVERYTHING_OK = 3

    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf  # Initially, loss is infinity

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return self.EVERYTHING_OK
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter == self.patience:
                return self.PATIENCE_ENDED
            return self.PATIENCE_STARTED
