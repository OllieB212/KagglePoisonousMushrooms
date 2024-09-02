import torch
from torchmetrics.metric import Metric
from EarlyStopping import EarlyStopping
from tqdm.notebook import tqdm
from torchmetrics.metric import Metric
import numpy as np
import pandas as pd


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

        if batch % batchStatusUpdate == 0:
            print(f"\tBatch: {batch}")

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
        for batch, (X, Y) in enumerate(dataLoader):

            # Put the data on the appropiate device:
            X = X.to(device)
            Y = Y.to(device)
            # Get the predicted Logits
            logits = model(X)

            # calculate the metric:
            metric(logits, torch.argmax(Y, 1))

    return metric.compute()


def train_model(
    model: torch.nn.Module,
    maxEpochs: int,
    modelSavePath: str,
    modelName: str,
    dataLoaderTrain: torch.utils.data.DataLoader,
    dataLoaderValid: torch.utils.data.DataLoader,
    lossFn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric: Metric,
    device: torch.device,
    earlyStoppingArgs=[],
    validateLogitActivation=None,
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
    for epoch in tqdm(range(0, maxEpochs)):

        trainLoss = train_epoch(
            model, dataLoaderTrain, lossFn, optimizer, device, batchStatusUpdate
        )

        validLoss = validate_epoch(model, dataLoaderValid, metric, device)

        print(f"Train Loss epoch {epoch}: {trainLoss}")
        print(f"Valid score epoch {epoch}: {validLoss}")

        if stopper(model, validLoss):
            stopper.restoreBestWeights(model)

            torch.save(
                model.state_dict(),
                f"{modelSavePath}/{modelName}.pth",
            )
            print(
                f"Stopped at epoch: {epoch}\nBest weights at epoch: {epoch-stopper.counter}"
            )
            return model.state_dict()

        torch.save(model.state_dict(), f"{modelSavePath}/{modelName}_latest.pth")

    torch.save(model.state_dict(), f"{modelSavePath}/{modelName}_UNSTOPPED_LATEST.pth")
    stopper.restoreBestWeights(model)
    torch.save(
        model.state_dict(), f"{modelSavePath}/{modelName}_UNSTOPPED_BEST_WEIGHTS.pth"
    )

    return model.state_dict()


def predict(
    model: torch.nn.Module,
    dataLoader: torch.utils.data.DataLoader,
    idArrPath,
    device: torch.device,
):
    """
    This function computes the following evaluation metrics on a given dataset: Accuracy, recall, confusion matrix, 1 vs rest confusion matrix.

    Parameters
    ----------
    model: The model to evaluate.

    dataLoader: Dataloader for the dataset to evaluate the model.

    resultSavePath: path to where the results shoul be saved.

    name of the results file.

    device: Device to run the model on.

    """
    df = pd.DataFrame(np.load(idArrPath), columns=["id"])
    arr = torch.Tensor().to(device)

    model.eval()
    with torch.inference_mode():
        for batch, X in tqdm(enumerate(dataLoader)):
            X = X.to(device)

            y_pred = model(X)
            arr = torch.cat((arr, y_pred), 0)

    df["class"] = pd.Series(torch.squeeze(arr, 1).cpu().numpy().astype(np.float32))

    return df


def train_model_libAUC(
    model: torch.nn.Module,
    maxEpochs: int,
    modelSavePath: str,
    modelName: str,
    dataLoaderTrain: torch.utils.data.DataLoader,
    dataLoaderValid: torch.utils.data.DataLoader,
    lossFn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric: Metric,
    device: torch.device,
    logitActivation=None,
    lrDecay=None,
    earlyStoppingArgs=[],
    batchStatusUpdate=500,
    seed=42,
    libAUC=False,
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
    for epoch in tqdm(range(0, maxEpochs)):

        if lrDecay is not None:
            if epoch in lrDecay:
                if libAUC:
                    optimizer.update_regularizer(decay_factor=10)
                else:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = 0.1 * param_group["lr"]

        trainLoss = train_epoch(
            model, dataLoaderTrain, lossFn, optimizer, device, batchStatusUpdate
        )

        validLoss = validate_epoch(
            model, dataLoaderValid, metric, device, logitActivation=logitActivation
        )

        print(f"Train Loss epoch {epoch}: {trainLoss}")
        print(f"Valid score epoch {epoch}: {validLoss}")

        if stopper(model, validLoss):
            stopper.restoreBestWeights(model)

            torch.save(
                model.state_dict(),
                f"{modelSavePath}/{modelName}.pth",
            )
            print(
                f"Stopped at epoch: {epoch}\nBest weights at epoch: {epoch-stopper.counter}"
            )
            return model.state_dict()

        torch.save(model.state_dict(), f"{modelSavePath}/{modelName}_latest.pth")

    torch.save(model.state_dict(), f"{modelSavePath}/{modelName}_UNSTOPPED_LATEST.pth")
    stopper.restoreBestWeights(model)
    torch.save(
        model.state_dict(), f"{modelSavePath}/{modelName}_UNSTOPPED_BEST_WEIGHTS.pth"
    )

    return model.state_dict()
