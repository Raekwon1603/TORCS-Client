# Torcs Machine Learning Driver

The machine learning driver is developed to control a race car on the Torcs simulation platform. In this repository, you can find the training algorithm in `build_pipeline/build_pipeline.py`. The data used for this is located in the `csv` folder. The details of how this data was collected are explained in the README inside the `data_pipeline` folder.

A working final version of the model can be found in the `models/final_model/` folder.

## Contents

- `Build_pipeline/` Contains the script used to train the model.
- `csv/` Contains the training data used for the model.
- `driver/` Contains the program used to run the model on the Torcs server.
- `models/` Contains other models trained outside the latest model.
- `models/final_model/`: Contains the most recent model.
- `notebooks/` Contains Jupyter notebooks used during the development of the model.
- `Dockerfile`: Contains the instructions for building the Docker image.
- `NeuralNet.py`: Contains classes and functions used by multiple scripts in the project.
- `NeuralNetSettings`: Contains the settings used for the model, including structure and the data fields used.

## Contributors

- [Raekwon Killop](https://github.com/raekwonkillop)
- [Raymond Blok](https://github.com/raymond-blok)
- [Arthur Struik](https://github.com/Matthino868)

## Using the Model

You can use the model by either building or pulling the Docker container. Below are the instructions for both methods.

### Method 1: Docker Build

1. Clone this repository to your local machine.
2. Navigate to the repository directory.
3. Run the following command to build the Docker container:

    ```bash
    docker build -t mldriver .
    ```

### Method 2: Docker Pull

You can also directly pull the prepared Docker image from Docker Hub.

1. Run the following command to pull the Docker image:

    ```bash
    docker pull raymondblok/mldriver
    ```

## Running the Container

Once you have built or pulled the Docker container, you can run it with:

```bash
docker run -it [raymondblok/]mldriver --hostname "torcs server ipaddress" --port "torcs server port"
