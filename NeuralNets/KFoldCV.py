import torch
from sklearn.model_selection import StratifiedKFold
import torch.utils
import torch.utils.data
from EarlyStopping import EarlyStopping
from torchmetrics.metric import Metric
from tqdm.notebook import tqdm
import pickle


class KFoldCV:

    def __init__(
        self,
        k: int,
        metric: Metric,
        modelClass: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        lossFn: torch.nn.Module,
        optimizerClass: torch.optim.Optimizer,
        device: torch.device,
        batchSize: int = 64,
        maxEpochs: int = 100,
        savePath: str = "",
        modelName: str = "model0",
        modelArgs: list = [],
        optimizerArgs: list = [],
        earlyStoppingArgs: list = [],
        useInitialWeights: torch.nn.Module = None,
        sampler=None,
        seed=42,
        batchStatusUpdate=500,
    ):

        self.metric = metric
        self.modelClass = modelClass
        self.maxEpochs = maxEpochs
        self.savePath = savePath
        self.modelName = modelName
        self.dataset = dataset
        self.lossFn = lossFn
        self.optimizerClass = optimizerClass
        self.device = device
        self.batchSize = batchSize
        self.modelArgs = modelArgs
        self.optimizerArgs = optimizerArgs
        self.earlyStoppingArgs = earlyStoppingArgs
        self.sampler = sampler
        self.seed = seed
        self.batchStatusUpdate = batchStatusUpdate
        self.useInitialWeights = useInitialWeights

        self.trainedModels = []
        self.validScores = []

        if seed is not None:
            self.kfold = StratifiedKFold(
                n_splits=k, shuffle=True, random_state=self.seed
            )
            self.splits = self.kfold.split(self.dataset.X, self.dataset.y)

        else:
            self.kfold = StratifiedKFold(n_splits=k, shuffle=False)
            self.splits = self.kfold.split(self.dataset.X, self.dataset.y)

    def crossValidate(self):

        for fold, (train_idx, valid_idx) in enumerate(self.splits):
            print(f"Fold: {fold + 1}")

            train_loader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=self.batchSize,
                sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            )
            valid_loader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=self.batchSize,
                sampler=torch.utils.data.SubsetRandomSampler(valid_idx),
            )

            model = self.modelClass(*self.modelArgs).to(self.device)

            if self.useInitialWeights is not None:
                model.load_state_dict(self.useInitialWeights)

            optimizer = self.optimizerClass(model.parameters(), *self.optimizerArgs)

            self.train_model(
                model=model,
                foldNum=fold,
                dataLoaderTrain=train_loader,
                dataLoaderValid=valid_loader,
                optimizer=optimizer,
                seed=42,
            )

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataLoader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
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
            X = X.to(self.device)
            Y = Y.to(self.device)

            # Forward pass:
            y_pred = model(X)

            # Calculate the loss:
            loss = self.lossFn(y_pred, Y)

            # Update training loss:
            trainLoss += loss.item() * Y.shape[0]
            nSamples += Y.shape[0]

            # Zero the optimizer:
            optimizer.zero_grad()

            # Backwards pass:
            loss.backward()

            # Improve model:
            optimizer.step()

            if batch % self.batchStatusUpdate == 0:
                print(f"\tBatch: {batch}")

        return trainLoss / nSamples

    def validate_epoch_metric(
        self,
        model: torch.nn.Module,
        dataLoader: torch.utils.data.DataLoader,
    ):
        """
        This function evaluates the model using a loss function. Used for validation

        Parameters
        ----------
        model: the model to be evaluated.

        dataLoader: The dataset to evaluate the model.

        LossFn: used to calculate the error on the dataset.

        """
        # Put the model in evaluation mode:
        model.eval()
        self.metric.reset()
        with torch.inference_mode():
            for batch, (X, Y) in enumerate(dataLoader):

                # Put the data on the appropiate device:
                X = X.to(self.device)
                Y = Y.to(self.device)

                # Get the predicted Logits
                y_pred = model(X)

                # calculate the metric:
                self.metric(y_pred, Y)

        return self.metric.compute()

    def train_model(
        self,
        model: torch.nn.Module,
        foldNum: int,
        dataLoaderTrain: torch.utils.data.DataLoader,
        dataLoaderValid: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        seed=42,
    ):
        """
        This function trains a model for given number of epochs.

        Parameters
        ----------
        model: The model that needs to be trained

        epochs: The maximum amount of epochs to train the model on (for early stopping).

        modelSavePath: The path to the directory where the model gets saved for each epoch

        modelName: The name of the model (Used for saving the model).

        dataLoaderTrain: The dataloader used to train the model

        dataLoaderValid: The dataloader of the validation data to evaluate the model (NOT test data).

        lossFn: Loss function to evalute the model

        optimizer: The optimizer used to improve the model

        metric: The torchmetric used for vaalidation

        device: The device to run the model on

        start: Used for naming purposes for saving the model if the model has undergone training before. Defaults to 0 (assumesmodel has not been trained yet).

        seed: Sets the random state of the model for reproducibility. Defaults to 42. NOTE: random state may not be excactly the same as CUDA has its own randomness on the graphics card.
        """

        trainData = {"trainLoss": [], "validScore": []}
        stopper = EarlyStopping(*self.earlyStoppingArgs)
        torch.manual_seed(seed)
        for epoch in tqdm(range(0, self.maxEpochs)):

            trainLoss = self.train_epoch(model, dataLoaderTrain, optimizer)

            trainData["trainLoss"].append(trainLoss)

            validScore = self.validate_epoch_metric(model, dataLoaderValid)
            trainData["validScore"].append(validScore)

            print(f"Train Loss epoch {epoch}: {trainLoss}")
            print(f"Valid Score epoch {epoch}: {validScore}")

            if stopper(model, validScore):
                stopper.restoreBestWeights(model)
                with open(
                    f"{self.savePath}/{self.modelName}_TRAINDATA_FOLD_{foldNum}.pickle",
                    "wb",
                ) as handle:
                    pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)

                torch.save(
                    model.state_dict(),
                    f"{self.savePath}/{self.modelName}_FOLD_{foldNum}.pth",
                )
                print(
                    f"Stopped at epoch: {epoch}\nBest weights at epoch: {epoch-stopper.counter}"
                )
                self.validScores.append(stopper.bestValid)
                return trainData

            # torch.save(model.state_dict(), f"{modelSavePath}/{modelName}_latest.pth")
        self.trainedModels.append(model)
        torch.save(
            model.state_dict(),
            f"{self.savePath}/{self.modelName}_FOLD_{foldNum}_UNSTOPPED.pth",
        )
        with open(
            f"{self.savePath}/{self.modelName}_data_unstopped.pickle", "wb"
        ) as handle:
            pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return trainData
