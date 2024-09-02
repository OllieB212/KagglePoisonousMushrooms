import torch
from torchmetrics.metric import Metric
from EarlyStopping import EarlyStopping
from torchmetrics.metric import Metric
import optuna
import numpy as np
import pandas as pd
import pickle


class PytorchOptuna:
    """
    This class serves as a wrapper for optuna making implementation quicker and more reliable.
    """

    def __init__(
        self,
        SavePath,
        create_model,
        objective,
        storage=None,
        sampler=None,
        pruner=None,
        study_name=None,
        direction="maximize",
        load_if_exists=False,
        directions=None,
    ):

        self.savePath = SavePath
        self.study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            direction=direction,
            load_if_exists=load_if_exists,
            directions=directions,
        )

        self.model = None
        self.create_model_ = create_model
        self.objective_ = objective
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(self, trial):

        self.model = self.create_model_(trial)

        return self.objective_(trial, self.model)

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:

            with open(
                f"{self.savePath}/BEST_MODEL_PARAMETERS.pickle",
                "wb",
            ) as handle:
                pickle.dump(study.best_trial, handle, protocol=pickle.HIGHEST_PROTOCOL)

            torch.save(
                self.model.state_dict(),
                f"{self.savePath}/BEST_MODEL_WEIGHTS.pth",
            )

    def optimize(
        self,
        n_trials=None,
        timeout=None,
        n_jobs=1,
        catch=(),
        gc_after_trial=False,
        show_progress_bar=True,
    ):
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=catch,
            callbacks=[self.callback],
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )


def train_epoch(
    model: torch.nn.Module,
    dataLoader: torch.utils.data.DataLoader,
    lossFn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batchStatusUpdate,
):
    """
    This function trains a model for one epoch

    Parameters
    ----------
    model: The model that needs to be trained

    dataLoader: the dataloader used to train the model

    lossFn: Loss function to evalute the model

    optimizer: The optimizer used to improve the model

    device: THe device to run the model on
    """

    # Enter training mode:
    model.train()
    trainLoss = 0  # initialize training loss to zero at start of each epoch
    nSamples = 0
    for batch, (X, Y) in enumerate(dataLoader):
        # Put the data on the appropiate device:
        X = X.to(device)
        Y = Y.to(device)

        # Forward pass:
        y_pred = model(X)

        # Calculate the loss:
        loss = lossFn(y_pred, Y)

        # Update training loss:
        trainLoss += loss.item() * Y.shape[0]
        nSamples += Y.shape[0]

        # Zero the optimizer:
        optimizer.zero_grad()

        # Backwards pass:
        loss.backward()

        # Improve model:
        optimizer.step()

        # if batch % batchStatusUpdate == 0:
        #     print(f"\tBatch: {batch}")

    return trainLoss / nSamples


def validate_epoch(
    model: torch.nn.Module,
    dataLoader: torch.utils.data.DataLoader,
    metric: Metric,
    device: torch.device,
    logitActivation=None,
):
    """
    This function evaluates the model. Used for validation

    Parameters
    ----------
    model: the model to be evaluated.

    dataLoader: The dataset to evaluate the model.

    LossFn: used to calculate the error on the dataset.

    """
    # Put the model in evaluation mode:
    model.eval()
    with torch.inference_mode():
        for X, Y in dataLoader:

            # Put the data on the appropiate device:
            X = X.to(device)
            Y = Y.to(device)
            # Get the predicted Logits
            logits = model(X)

            if logitActivation is not None:
                y_pred = logitActivation(logits)
            else:
                y_pred = logits

            # calculate the metric:
            metric(y_pred, Y)

    return metric.compute()


evaluate_model = validate_epoch


def train_model(
    model: torch.nn.Module,
    maxEpochs: int,
    dataLoaderTrain: torch.utils.data.DataLoader,
    dataLoaderValid: torch.utils.data.DataLoader,
    lossFn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric: Metric,
    device: torch.device,
    trial,
    logitActivation=None,
    earlyStoppingArgs=[],
    batchStatusUpdate=500,
    seed=42,
):
    """
    This function trains a model for given number of epochs.

    Parameters
    ----------
    model: The model that needs to be trained

    epochs: The amount of epochs to train the model on.

    modelSavePath: The path to the directory where the model gets saved for each epoch

    modelName: The name of the model (Used for saving the model).

    dataLoaderTrain: The dataloader used to train the model

    dataLoaderValid: The dataloader of the validation data to evaluate the model (NOT test data).

    lossFn: Loss function to evalute the model

    optimizer: The optimizer used to improve the model

    device: THe device to run the model on

    start: Used for naming purposes for saving the model if the model has undergone training before. Defaults to 0 (assumesmodel has not been trained yet).

    seed: Sets the random state of the model for reproducibility. Defaults to 42. NOTE: random state may not be excactly the same as CUDA has its own randomness on the graphics card.
    """

    stopper = EarlyStopping(*earlyStoppingArgs)
    torch.manual_seed(seed)
    for epoch in range(0, maxEpochs):

        trainLoss = train_epoch(
            model, dataLoaderTrain, lossFn, optimizer, device, batchStatusUpdate
        )

        validLoss = validate_epoch(
            model, dataLoaderValid, metric, device, logitActivation=logitActivation
        )

        # print(f"Train Loss epoch {epoch}: {trainLoss}")
        # print(f"Valid score epoch {epoch}: {validLoss}")

        if stopper(model, validLoss):
            break

        trial.report(validLoss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    stopper.restoreBestWeights(model)

    return validLoss
