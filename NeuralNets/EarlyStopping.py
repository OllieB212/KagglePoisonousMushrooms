from copy import deepcopy
import torch


class EarlyStopping:

    def __init__(self, tol=1e-3, patientce=15, minimize=False):
        """
        This Class implements early stopping.

        Parameters
        ----------
        tol: The tolerance (minimum increase in performance to reset counter).
        patientce: Maximum value of the counter before stopping.
        minimize: True for minimizing the validation value, otherwise False when intending to maximize the validation value. These values depend on the metric used for validation.

        """
        self.tol = tol
        self.patientce = patientce
        self.counter = 0
        self.bestModel = None
        self.bestValid = None
        self.minimize = minimize

    def __call__(self, model, valid):

        if self.bestValid is None:
            self.bestValid = valid
            self.bestModel = deepcopy(model.state_dict())

        elif self.minimize and self.tol <= self.bestValid - valid:
            self.counter = 0
            self.bestValid = valid
            self.bestModel = deepcopy(model.state_dict())

        elif (not self.minimize) and (self.tol <= valid - self.bestValid):
            self.counter = 0
            self.bestValid = valid
            self.bestModel = deepcopy(model.state_dict())

        else:
            self.counter += 1

            if self.counter == self.patientce:
                print(f"Early stopping. counter: {self.counter}")

                return True

        return False

    def restoreBestWeights(self, model):

        model.load_state_dict(self.bestModel)
        print("Best weights restored.")
