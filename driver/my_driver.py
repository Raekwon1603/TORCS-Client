import sys
from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import time
from pathlib import Path

# Ensure the custom module path is included
sys.path.append("..")
from NeuralNet import loadModel
import NeuralNetSettings as settings

class MyDriver(Driver):
    def __init__(self, logdata=True):
        super().__init__(logdata)
        self.stuck_count = 0
        self.is_stuck = False
        self.reverse_count = 0
        self.last_reverse_end_time = None  # Timer for last reverse end time

        # Define the model parameters and load the model
        modelName = "final_model"
        modelPath = Path.cwd().parent / Path("models")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.scaler, _ = loadModel(modelPath / modelName, settings.layerSettings["inputLayerSize"], settings.layerSettings["outputLayerSize"], settings.layerSettings["HiddenLayerSizes"])
        self.model.eval()

    def drive(self, carstate: State) -> Command:
        # Check if car stuck in a wall
        if 0 < carstate.speed_x <= 0.5 and len([edge for edge in carstate.distances_from_edge if -1 <= edge <= 5]) >= 9 and carstate.current_lap_time > 10.0:
            self.stuck_count += 5
        # Check if car stuck in another opponent
        elif -1.0 < carstate.speed_x <= 1 and len([opponents for opponents in carstate.opponents if 0 <= opponents <= 6]) >= 1 and -2.5 <= carstate.angle <= 2.5 and carstate.current_lap_time > 10.0:
            self.stuck_count += 5
        else:
            self.stuck_count = max(0, self.stuck_count - 1)

        # If car is stuck for too long
        if self.stuck_count >= 50:
            self.is_stuck = True

        # Delay before reversing again
        if self.last_reverse_end_time and time.time() - self.last_reverse_end_time < 5:
            self.is_stuck = False
            self.stuck_count = 0
            self.reverse_count = 0

        if self.is_stuck:
            self.reverse_count += 1
            command = Command()
            command.gear = -1  # Reverse gear
            command.accelerator = 1.0  # Increase reverse speed
            command.brake = 0  # Release brake

            if self.reverse_count > 120 and len([edge for edge in carstate.distances_from_edge if edge > 0]) >= 6 or carstate.speed_x < -13.0:
                self.is_stuck = False
                self.stuck_count = 0
                self.reverse_count = 0
                self.last_reverse_end_time = time.time()  # Set the last reverse end time

            return command

        # Normal driving
        speed_x = carstate.speed_x * 3.6
        speed_y = carstate.speed_y * 3.6
        speed_z = carstate.speed_z * 3.6
        angle = carstate.angle * (3.1415 / 180)
        track_position = carstate.distance_from_center
        track_edges = carstate.distances_from_edge[:19]
        opps = carstate.opponents

        input_data = [speed_x, speed_y, angle, track_position] + list(track_edges) + [speed_z] + list(opps)
        
        input_df = pd.DataFrame([input_data], columns=settings.features['inputs'])
        input_df = self.scaler.transform(input_df)
        input_tensor = torch.tensor(input_df, dtype=torch.float32).to(self.device)

        # Get output from model
        with torch.no_grad():
            output = self.model(input_tensor)
        output = output.cpu().numpy().flatten()
        command = Command()
        
        # Calculate gear
        command.gear = shifter(carstate.gear, carstate.rpm)
        command.brake = output[0]
        command.accelerator = output[1]
        command.steering = output[2]
        return command

def shifter(gear, rpm):
    '''
    Calculates gear based on previous `gear` and `rpm`
    '''
    if gear == 0:
        return 1
    if rpm > 8000 and gear < 4:
        return gear + 1
    elif rpm < 2000 and gear > 1:
        return gear - 1
    elif rpm < 0:
        return 1
    else:
        return gear
