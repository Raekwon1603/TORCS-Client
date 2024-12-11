features = {
  "inputs": list([' SpeedX', ' SpeedY', 'Angle', 'TrackPosition', 
                    ' Track_1','Track_2','Track_3','Track_4','Track_5','Track_6','Track_7','Track_8','Track_9','Track_10','Track_11','Track_12','Track_13','Track_14','Track_15','Track_16','Track_17','Track_18','Track_19' 
                    ,' SpeedZ',' Opponent_1', 'Opponent_2', 'Opponent_3', 'Opponent_4', 'Opponent_5', 'Opponent_6', 'Opponent_7', 'Opponent_8', 'Opponent_9', 'Opponent_10',
                    'Opponent_11', 'Opponent_12', 'Opponent_13', 'Opponent_14', 'Opponent_15', 'Opponent_16', 'Opponent_17', 'Opponent_18', 'Opponent_19', 'Opponent_20',
                    'Opponent_21', 'Opponent_22', 'Opponent_23', 'Opponent_24', 'Opponent_25', 'Opponent_26', 'Opponent_27', 'Opponent_28', 'Opponent_29', 'Opponent_30',
                    'Opponent_31', 'Opponent_32', 'Opponent_33', 'Opponent_34', 'Opponent_35', 'Opponent_36'
                    ]),
  "outputs": list(['Braking', ' Acceleration', 'Steering'])
}


layerSettings = {
    "inputLayerSize": len(features["inputs"]),
    "outputLayerSize": len(features["outputs"]),
    "HiddenLayerSizes": [256, 128, 64, 32, 16]
}
