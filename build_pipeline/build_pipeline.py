import torch
import os
import re
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse as ap
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import NeuralNetSettings as settings
from NeuralNet import Net, CustomDataset, loadModel, saveModel

DEFAULT_MODEL_PATH = "..\\models"
DEFAULT_DATA_PATH = "..\\csv\\data.csv"
GENERIC_MODEL_FILE_NAME = "model_epoch_"
GENERIC_SCALAR_FILE_NAME = "scalar_epoch_"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelRegexFilter = re.compile(GENERIC_MODEL_FILE_NAME + r"(?P<epoch>\d+)")
scalerRegexFilter = re.compile(GENERIC_SCALAR_FILE_NAME + r"(?P<epoch>\d+)")

def readCSV(dataPath: Path) -> pd.DataFrame:
    '''
    Read every csv file from `datapath` and concatenate them into a `dataframe`
    '''
    if dataPath.is_file():
        return pd.read_csv(dataPath, sep=",")
    if dataPath.exists():
        files = [x for x in dataPath.rglob("*.csv")]

        df = pd.DataFrame()
        for (i, file) in enumerate(files):
            dfTemp = pd.read_csv(file, sep=",")
            # logging.info(f"Loading: {file.name} with {dfTemp.shape[0]} rows")
            print(f"Loading data: {i}/{len(files)}", end='\r')
            df = pd.concat([df,dfTemp],ignore_index=True)

        logging.info(f"Total data frame has: {df.shape[0]} rows")
        return df

def calculate_accuracy(true_labels, predictions, threshold=0.1):
    '''
    Calculate the accuracy of the model
    '''
    correct = np.abs(true_labels - predictions) < threshold
    accuracy = correct.mean() * 100
    return accuracy

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    # Parse every commandline arguments
    parser = ap.ArgumentParser()
    parser.add_argument("model", action="store", type=str, help="Name of the model to be trained")
    parser.add_argument("-p", "--model-path", action="store", type=str, dest="model_path", help="Custom destination to write model to", default=DEFAULT_MODEL_PATH, nargs="?")
    parser.add_argument("-d", "--data-path", action="store", type=str, dest="data_path", help="Custom destination to get the training data from", default=DEFAULT_DATA_PATH, nargs="?")
    parser.add_argument("-s", "--save-epoch", action="store", type=int, dest="save_epoch", help="Amount of epochs inbetween saves", default=10, nargs="?")
    parser.add_argument("-e", "--epochs", action="store", type=int, dest="epochs", help="Amount of epochs", default=100, nargs="?")
    parser.add_argument("-l", "--file-limit", action="store", type=int, dest="file_limit", help="max files number of files used before deleting the old ones", default=5, nargs="?")
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--no-graph', dest='graph', action='store_false')
    parser.set_defaults(graph=True)

    args = parser.parse_args()

    modelName = args.model

    dataPath = Path(args.data_path)
    matlab = args.graph

    modelPath = Path(args.model_path)
    amountOfEpochs = args.save_epoch
    epochs = args.epochs
    fileLimit = args.file_limit

    logging.info(f"Starting training using {device}")
    initial_memory = torch.cuda.memory_allocated()

    # Load in dataframe
    try:
        logging.info(f"Loading data...")
        df = readCSV(dataPath)
    except Exception as e:
        logging.error(e)
        exit()

    # Drop rows with NULL values
    dfSelected = df.copy()
    dfSelected.dropna(axis=0)
    
    # Select input and output features
    input = dfSelected[settings.features["inputs"]]
    output = dfSelected[settings.features["outputs"]]

    logging.info(f"Loaded {settings.layerSettings['inputLayerSize']} input features.")
    logging.info(f"Loaded {settings.layerSettings['outputLayerSize']} input features.")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)

    # Load in previously trained model
    specificModelPath = modelPath / modelName
    net, scaler, oldEpoch = loadModel(specificModelPath, settings.layerSettings["inputLayerSize"], settings.layerSettings["outputLayerSize"], settings.layerSettings["HiddenLayerSizes"])

    # # Standardize scaler features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)


    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # User L1 as loss function and Adam as optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training loop
    loss_data = []
    test_losses = []

    print(f"Model name: {modelName}")
    print(f"Training data used: {dataPath.name}")
    if matlab:
        # Show graph with loss and accuracy
        fig, axis = plt.subplots(2)
        axis[0].set_title("training loss data")
        axis[1].set_title("accuracy")
        axis[0].set(ylabel='Loss')
        axis[1].set(xlabel='Epoch', ylabel='accuracy in percentage')
        fig.tight_layout()

        plt.ion()
        lossDataGraph = axis[0].plot([0])[0]
        testLossesGraph = axis[0].plot([0])[0]
        accuracyGraph = axis[1].plot([0])[0]
        axis[0].legend([lossDataGraph, testLossesGraph], ["test data", "training data"])
        plt.draw()
        axis[0].set_xlim(oldEpoch, epochs+oldEpoch)
        axis[1].set_xlim(oldEpoch, epochs+oldEpoch) 

    LossHistory = {
        "loss_data": [],
        "test_losses": [],
        "accuracy": []
        }
    
    current_memory = torch.cuda.memory_allocated()

    # Calculate the memory allocated on GPU
    allocated_memory = current_memory - initial_memory
    logging.info(f"Memory allocated on GPU: {allocated_memory//1_048_576} MBytes")
    for epoch in range(1, epochs+1):
        if epoch != 0 and epoch % amountOfEpochs == 0:
            saveModel(net, scaler, specificModelPath, epoch+oldEpoch, fileLimit)
        if matlab:
            testLossesGraph.remove()
            testLossesGraph = axis[0].plot(range(oldEpoch, oldEpoch+epoch-1), LossHistory["test_losses"], color="b")[0]
            lossDataGraph.remove()
            lossDataGraph = axis[0].plot(range(oldEpoch, oldEpoch+epoch-1), LossHistory["loss_data"], color="r")[0]
            accuracyGraph.remove()
            accuracyGraph = axis[1].plot(range(oldEpoch, oldEpoch+epoch-1), LossHistory["accuracy"], color="g")[0]
            plt.draw()
            plt.pause(0.001)
        
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Epoch [{epoch+oldEpoch}/{epochs+oldEpoch}], Loss: {running_loss/len(train_loader)}                    ', end="\r")
        LossHistory["loss_data"].append(running_loss/len(train_loader))

        # Evaluation
        net.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            test_loss = 0
            predictions = []
            true_labels = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            predictions = np.array(predictions)
            true_labels = np.array(true_labels)

            LossHistory["accuracy"].append(calculate_accuracy(true_labels, predictions))
            LossHistory["test_losses"].append(test_loss/len(test_loader))        

        
    print("")

    # Save model and scaler
    saveModel(net, scaler, specificModelPath, epoch+oldEpoch, fileLimit)
    if matlab:
        lossDataGraph.remove()
        lossDataGraph = axis[0].plot(range(oldEpoch, oldEpoch+epochs), LossHistory["loss_data"], color="r")[0]
        testLossesGraph.remove()
        testLossesGraph = axis[0].plot(range(oldEpoch, oldEpoch+epochs), LossHistory["test_losses"], color="b")[0]
        accuracyGraph.remove()
        accuracyGraph = axis[1].plot(range(oldEpoch, oldEpoch+epochs), LossHistory["accuracy"], color="g")[0]
        plt.draw()
        plt.pause(0.001)
        plt.show(block=True)
        plt.close()

    print("Done")