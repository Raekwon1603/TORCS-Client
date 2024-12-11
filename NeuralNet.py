import torch
import os
import re
import pickle
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DEFAULT_MODEL_PATH = "..\\models"
DEFAULT_DATA_PATH = "..\\csv\\data.csv"
GENERIC_MODEL_FILE_NAME = "model_epoch_"
GENERIC_SCALAR_FILE_NAME = "scalar_epoch_"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelRegexFilter = re.compile(GENERIC_MODEL_FILE_NAME + r"(?P<epoch>\d+)")
scalerRegexFilter = re.compile(GENERIC_SCALAR_FILE_NAME + r"(?P<epoch>\d+)")


class Net(torch.nn.Module):
    fc: torch.nn.ModuleList

    def __init__(self, input_size: int, num_classes: int, hiddenLayersSize: list[int]) -> None:
        super(Net, self).__init__()
        self.fc = torch.nn.ModuleList()
        prevLayerSize = input_size
        for hiddenLayerSize in hiddenLayersSize+[num_classes]:
            self.fc.extend([torch.nn.Linear(prevLayerSize, hiddenLayerSize)])
            prevLayerSize = hiddenLayerSize
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.fc[:-1]:
            x = torch.relu(layer(x))
        x = self.fc[-1](x)
        return x
    

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels) -> None:
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels.values, dtype=torch.float32).to(device)
        self.inputs = torch.tensor(features, dtype=torch.float32).to(device)   
          
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.features[idx], self.labels[idx]
    

def saveModel(model, scaler, path: Path, epoch, fileLimit):
    '''
    Save model and scaler to `path`. Remove old models and scalers if `noOfFiles > fileLimit` 
    '''
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), path / (GENERIC_MODEL_FILE_NAME + str(epoch) + ".pt"))
    with open(path /  (GENERIC_SCALAR_FILE_NAME+str(epoch) + ".pkl"), "wb") as file:
        pickle.dump(scaler, file)
    
    files = [f for f in path.iterdir() if f.is_file()]
    if len(files) > 0:
        modelEpochList = [int(modelRegexFilter.match(file.name).groups("epoch")[0])  for file in files if modelRegexFilter.match(file.name)]
        scalarEpochList = [int(scalerRegexFilter.match(file.name).groups("epoch")[0])  for file in files if scalerRegexFilter.match(file.name)]
        modelEpochList.sort(reverse=True)
        scalarEpochList.sort(reverse=True)
        for epoch in modelEpochList[fileLimit:]:
            os.remove(path / (GENERIC_MODEL_FILE_NAME + str(epoch) + ".pt"))
            os.remove(path / (GENERIC_SCALAR_FILE_NAME+str(epoch) + ".pkl"))
    

def loadModel(path: Path, input_size, num_classes, hiddenLayers):
    '''
    Create `Net`with`input_size` and `num_classes`.
    Load model and scaler from `path`. 
    '''
    model = Net(input_size, num_classes, hiddenLayers)
    scaler = StandardScaler()
    epoch = 0
    if os.path.exists(path):
        # files = os.listdir(path)
        # files = [f for f in files if os.path.isfile(path+'/'+f)]
        files = [f for f in path.iterdir() if f.is_file()]
        if len(files) > 0:
            modelEpochList = [int(modelRegexFilter.match(file.name).groups("epoch")[0])  for file in files if modelRegexFilter.match(file.name)]
            scalarEpochList = [int(scalerRegexFilter.match(file.name).groups("epoch")[0])  for file in files if scalerRegexFilter.match(file.name)]
            modelEpochList.sort(reverse=True)
            scalarEpochList.sort(reverse=True)
            if modelEpochList[0] != scalarEpochList[0]:
                for e in modelEpochList:
                    if e in scalarEpochList:
                        epoch = e
            else:
                epoch = modelEpochList[0]

            modelPath = path  / (GENERIC_MODEL_FILE_NAME + str(epoch) + ".pt")
            scalarPath = path / (GENERIC_SCALAR_FILE_NAME+str(epoch) + ".pkl")

            with open(modelPath, "rb") as file:
                model.load_state_dict(torch.load(file, map_location=device))

            with open(scalarPath, "rb") as file:
                scaler = pickle.load(file)
    else:
        # TODO foutmelding
        logging.info(f"No model named: {path.stem} found creating new model.")

    return model.to(device), scaler, epoch